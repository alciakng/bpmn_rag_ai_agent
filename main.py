
from config.analytics import track
from ui.free_use_manager import (
    free_questions_exhausted,
    user_supplied_openai_key_unavailable,
    decrement_free_questions,
)
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from streamlit_feedback import streamlit_feedback
from ui.constants import TITLE

from agent.hybid_graph_agent import HybridGraphAgent
from agent.dto.query_dto import QueryResult

import logging
import streamlit as st

# Logging options
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Anonymous Session Analytics
if "SESSION_ID" not in st.session_state:
    # Track method will create and add session id to state on first run
    track("rag_demo", "appStarted", {})

# LangChain caching to reduce API calls
set_llm_cache(InMemoryCache())

# App title
st.markdown(TITLE, unsafe_allow_html=True)

# Define message placeholder and emoji feedback placeholder
placeholder = st.empty()
emoji_feedback = st.empty()
user_placeholder = st.empty()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "content": f"본 프로젝트는 ARIS 모델 데이터기반 Business 업무 프로세스 최적화 AI-Agent 개발의 POC AI-Agent 입니다.",
        },
        # {"role": "ai", "content": f"""This the schema in which the EDGAR filings are stored in Neo4j: \n <img style="width: 70%; height: auto;" src="{SCHEMA_IMG_PATH}"/>"""},
        # {"role": "ai", "content": f"""This is how the Chatbot flow goes: \n <img style="width: 70%; height: auto;" src="{LANGCHAIN_IMG_PATH}"/>"""}
    ]

# Display chat messages from history on app rerun
with placeholder.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# User input - switch between sidebar sample quick select or actual user input. Clunky but works.
if free_questions_exhausted() and user_supplied_openai_key_unavailable():
    st.warning(
        "Thank you for trying out the Neo4j Rag Demo. Please input your OpenAI Key in the sidebar to continue asking questions."
    )
    st.stop()

if "sample" in st.session_state and st.session_state["sample"] is not None:
    user_input = st.session_state["sample"]
else:
    user_input = st.chat_input(
        placeholder="Ask question on the Process", key="user_input"
    )

@st.cache_resource
def initialize_rag_agent():
    """RAG 에이전트 초기화 (앱 시작시 한번만 실행)"""
    return HybridGraphAgent(
        neo4j_uri=st.secrets["NEO4J_URI"],
        neo4j_user=st.secrets["NEO4J_USERNAME"],
        neo4j_password=st.secrets["NEO4J_PASSWORD"], 
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

def handle_agent_response(rag_agent, user_input):
    try:
        # 에이전트 호출
        agent_response = rag_agent.query(user_input)
        
        # QueryResult 객체인지 확인
        if isinstance(agent_response, QueryResult):
            # 사용자 상호작용이 필요한 경우
            if agent_response.requires_user_interaction:
                st.warning(agent_response.interaction_message)
                return
            
            content = agent_response.answer
            # 일반 답변 표시
            st.markdown(agent_response.answer)

            track(
                "rag_demo", "ai_response", {"type": "rag_agent", "answer": content}
            )

            new_message = {"role": "ai", "content": content}
            st.session_state.messages.append(new_message)

            decrement_free_questions()
            
            # 이미지가 있는 경우 표시
            if agent_response.image_path:
                try:
                    # PDF URL인 경우 직접 링크 제공
                    if agent_response.image_path.startswith('http'):
                        st.markdown(f"📋 [프로세스 다이어그램 보기]({agent_response.image_path})")
                        # 또는 iframe으로 PDF 임베드
                        st.components.v1.iframe(agent_response.image_path, height=600)
                    else:
                        # 로컬 파일인 경우
                        st.image(agent_response.image_path)
                except Exception as e:
                    st.error(f"이미지 로드 실패: {e}")
            
            # 디버깅 정보 (선택적)
            if st.checkbox("상세 정보 보기"):
                st.json({
                    "query_type": agent_response.query_type.value,
                    "candidate_nodes_count": len(agent_response.candidate_nodes),
                    "has_image": bool(agent_response.image_path)
                })
        
        else:
            # 예상치 못한 응답 타입
            st.warning(f"예상치 못한 응답 타입: {type(agent_response)}")
            st.text(str(agent_response))
            
    except Exception as e:
        st.error(f"에이전트 처리 중 오류: {e}")


if user_input:
    with user_placeholder.container():
        track("rag_demo", "question_submitted", {"question": user_input})
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("ai"):

            # Agent response
            with st.spinner("..."):

                message_placeholder = st.empty()
                thought_container = st.container()

                # For displaying chain of thought - this callback handler appears to only works with the deprecated initialize_agent option (see rag_agent.py)
                # st_callback = StreamlitCallbackHandler(
                #   parent_container= thought_container,
                #   expand_new_thoughts=False
                # )
                # StreamlitCcallbackHandler api doc: https://api.python.langchain.com/en/latest/callbacks/langchain_community.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html
                rag_agent = initialize_rag_agent()

                handle_agent_response(rag_agent, user_input)

                decrement_free_questions()

    # Reinsert user chat input if sample quick select was previously used.
    if "sample" in st.session_state and st.session_state["sample"] is not None:
        st.session_state["sample"] = None
        user_input = st.chat_input(
            placeholder="Ask question on the Process", key="user_input"
        )

    emoji_feedback = st.empty()

# Emoji feedback
with emoji_feedback.container():
    feedback = streamlit_feedback(feedback_type="thumbs")
    if feedback:
        score = feedback["score"]
        last_bot_message = st.session_state["messages"][-1]["content"]
        track(
            "rag_demo",
            "feedback_submitted",
            {"score": score, "bot_message": last_bot_message},
        )
