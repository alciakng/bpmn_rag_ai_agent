
from datetime import timedelta
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.dto.process_state import ProcessState
from agent.dto.query_dto import QueryContext, QueryResult
from agent.entity.graph_node import GraphNode
from agent.enums.enums import QueryType
from agent.process_manager import ProcessManager
from utils.helpers import load_cfg
from utils.minio_utils import MinIOClient

import json
import uuid
import openai
import numpy as np
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import redis
from enum import Enum
from neo4j.exceptions import Neo4jError
import logging
import traceback


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridGraphAgent:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 openai_api_key: str, redis_host: str = "localhost", 
                 redis_port: int = 6379, cfg_file: str = "config.yaml"):
        """
        BPMN Hybrid Graph RAG 초기화
        """
        logger.info("=== HybridGraphAgent 초기화 시작 ===")
        
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info(f"Neo4j 연결 성공: {neo4j_uri}")
        except Exception as e:
            logger.error(f"Neo4j 연결 실패: {e}")
            raise
        
        try:
            openai.api_key = openai_api_key
            logger.info("OpenAI API 키 설정 완료")
        except Exception as e:
            logger.error(f"OpenAI API 키 설정 실패: {e}")
            raise
        
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, password="redis_password", decode_responses=False)
            # Redis 연결 테스트
            self.redis_client.ping()
            logger.info(f"Redis 연결 성공: {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise
        
        self.session_id = str(uuid.uuid4())
        logger.info(f"세션 ID 생성: {self.session_id}")
        logger.info("=== HybridGraphAgent 초기화 완료 ===\n")

    def close(self):
        """연결 종료"""
        logger.info("=== 에이전트 연결 종료 ===")
        self.driver.close()
        self.redis_client.close()

    def _contains_hangul(self, s: str) -> bool:
        return any('\uac00' <= ch <= '\ud7a3' for ch in s)

    def _ensure_db(self):
        return {"database": "neo4j"}

    def analyze_query_context(self, query: str) -> QueryContext:
        """향상된 질의 맥락 분석"""
        logger.info("=== 1단계: 질의 맥락 분석 시작 ===")
        logger.info(f"입력 질의: {query}")
        
        try:
            process_manager = ProcessManager(self.redis_client, self.session_id)
            
            # 1. 기본 질의 타입 분류
            logger.info("1-1. 질의 타입 분류 시작")
            query_type = self._classify_basic_query_type(query)
            logger.info(f"질의 타입 분류 결과: {query_type.value}")
            
            # 2. 이미지 요청 여부 확인
            logger.info("1-2. 이미지 요청 여부 확인")
            image_requested = self._check_image_request(query)
            logger.info(f"이미지 요청 여부: {image_requested}")
            
            # 3. 프로세스명 추출
            logger.info("1-3. 프로세스명 추출 시작")
            mentioned_process = process_manager.extract_process_name(query)
            logger.info(f"추출된 프로세스명: {mentioned_process}")
            
            # 4. 현재 프로세스 상태 확인
            logger.info("1-4. 프로세스 상태 확인")
            current_state = process_manager.get_current_state()
            current_process = process_manager.get_current_process()
            logger.info(f"현재 프로세스 상태: {current_state}")
            logger.info(f"현재 확정된 프로세스: {current_process}")
            
            # 5. 프로세스 처리 로직
            logger.info("1-5. 프로세스 처리 로직 실행")
            
            context = None
            if mentioned_process:
                logger.info(f"명시적 프로세스 언급 발견: {mentioned_process}")
                if current_process and mentioned_process != current_process['name']:
                    logger.info("프로세스 변경 요청 감지")
                    interaction_msg = process_manager.handle_process_change(mentioned_process)
                    context = QueryContext(
                        query_type=query_type,
                        process_name=mentioned_process,
                        process_action="confirm_change",
                        image_requested=image_requested,
                        requires_user_interaction=True,
                        interaction_message=interaction_msg
                    )
                else:
                    logger.info("새 프로세스 설정 또는 동일 프로세스 재확인")
                    context = QueryContext(
                        query_type=query_type,
                        process_name=mentioned_process,
                        process_action="search_new",
                        image_requested=image_requested,
                        requires_user_interaction=False,
                        interaction_message=None
                    )
            else:
                logger.info("프로세스 명시 없음 - 상태별 처리")
                if current_state == ProcessState.MULTIPLE_FOUND:
                    logger.info("여러 프로세스 후보 존재 - 사용자 선택 필요")
                    candidates = process_manager.get_process_candidates()
                    candidate_list = "\n".join([f"{i+1}. {p['name']}" for i, p in enumerate(candidates)])
                    context = QueryContext(
                        query_type=query_type,
                        process_name=None,
                        process_action="select_from_candidates",
                        image_requested=image_requested,
                        requires_user_interaction=True,
                        interaction_message=f"어떤 프로세스를 분석하시겠습니까?\n{candidate_list}\n번호를 선택해주세요."
                    )
                elif current_process:
                    logger.info("기존 확정 프로세스 사용")
                    context = QueryContext(
                        query_type=query_type,
                        process_name=current_process['name'],
                        process_action="use_current",
                        image_requested=image_requested,
                        requires_user_interaction=False,
                        interaction_message=None
                    )
                else:
                    logger.info("프로세스 미설정 - 프로세스 명시 요청")
                    context = QueryContext(
                        query_type=query_type,
                        process_name=None,
                        process_action="search_new",
                        image_requested=image_requested,
                        requires_user_interaction=True,
                        interaction_message="분석할 프로세스를 명시해주세요."
                    )
            
            logger.info(f"맥락 분석 최종 결과:")
            logger.info(f"  - 처리 액션: {context.process_action}")
            logger.info(f"  - 사용자 상호작용 필요: {context.requires_user_interaction}")
            logger.info("=== 1단계: 질의 맥락 분석 완료 ===\n")
            
            return context
            
        except Exception as e:
            logger.error(f"질의 맥락 분석 중 오류: {e}")
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            raise

    def _classify_basic_query_type(self, query: str) -> QueryType:
        """기본 질의 타입 분류"""
        logger.debug(f"질의 타입 분류 - 입력: {query}")
        
        classification_prompt = f"""
다음 사용자 질의를 분석하여 BPMN 프로세스 관련 질의 타입을 분류해주세요.

**질의:** {query}

**분류 기준:**
1. **FLOW_OVERVIEW**: BPMN 프로세스의 전체적인 흐름, 개요, 단계별 설명, 기능 설명, 역할과 책임 등을 묻는 질의
2. **PATH_FINDING**: 특정 노드에서 다른 노드까지의 경로, 연결 방법, 분기 조건, 라우팅을 묻는 질의  
3. **GENERAL**: 위 두 카테고리에 해당하지 않는 일반적인 정보 요청, 특정 개념 설명 등

**출력 형식:** FLOW_OVERVIEW, PATH_FINDING, GENERAL 중 하나만 출력
"""
        
        try:
            logger.debug("OpenAI API 호출 - 질의 타입 분류")
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"LLM 분류 결과: {result}")
            
            if "FLOW_OVERVIEW" in result:
                return QueryType.FLOW_OVERVIEW
            elif "PATH_FINDING" in result:
                return QueryType.PATH_FINDING
            else:
                return QueryType.GENERAL
                
        except Exception as e:
            logger.error(f"질의 타입 분류 오류: {e}")
            return QueryType.GENERAL

    def _check_image_request(self, query: str) -> bool:
        """이미지/다이어그램 요청 여부 확인"""
        image_keywords = ['이미지', '그림', '다이어그램', '도표', '차트', '시각화', '보여줘', '그려줘']
        result = any(keyword in query for keyword in image_keywords)
        logger.debug(f"이미지 키워드 검사 결과: {result}, 검사된 키워드: {image_keywords}")
        return result

    def two_stage_vector_search(self, query_context: QueryContext, 
                               avg_vec: np.ndarray) -> List[GraphNode]:
        """2단계 벡터 검색: 프로세스 -> 노드"""
        logger.info("=== 3단계: 2단계 벡터 검색 시작 ===")
        logger.info(f"검색 컨텍스트: {query_context.process_action}")
        
        process_manager = ProcessManager(self.redis_client, self.session_id)
        candidate_nodes = []
        
        with self.driver.session() as session:
            try:
                # 1단계: 프로세스 검색 (필요시)
                target_process_key = None
                
                if query_context.process_action == "search_new" and query_context.process_name:
                    logger.info(f"3-1. 새 프로세스 검색: {query_context.process_name}")

                    # 프로세스명을 임베딩으로 변환
                    process_embedding, ko_vec, en_vec = self.get_query_embeddings(query_context.process_name)

                    processes = process_manager.search_processes(query_context.process_name, session, process_embedding)
                    logger.info(f"프로세스 검색 결과: {len(processes)}개 발견")
                    
                    for i, proc in enumerate(processes):
                        logger.info(f"  {i+1}. {proc['name']} (key: {proc['key']})")
                    
                    if len(processes) == 1:
                        logger.info("단일 프로세스 자동 확정")
                        process_manager.confirm_process(processes[0]["key"], processes[0]["name"])
                        target_process_key = processes[0]["key"]
                    elif len(processes) > 1:
                        logger.info("여러 프로세스 후보 저장 - 사용자 선택 대기")
                        process_manager.set_process_candidates(processes)
                        return []
                    else:
                        logger.warning(f"프로세스 '{query_context.process_name}'를 찾을 수 없습니다.")
                        return []
                        
                elif query_context.process_action == "use_current":
                    logger.info("3-1. 현재 확정된 프로세스 사용")
                    current_process = process_manager.get_current_process()
                    target_process_key = current_process["key"] if current_process else None
                    logger.info(f"사용할 프로세스 키: {target_process_key}")
                else:
                    logger.info("3-1. 프로세스 필터링 없이 전체 검색")
                    target_process_key = None
                
                # 2단계: 노드 검색
                logger.info("3-2. 노드 검색 시작")
                
                if target_process_key:
                    if query_context.query_type == QueryType.FLOW_OVERVIEW:
                        logger.info("전체 플로우 요청 - 프로세스 내 모든 노드 조회")
                        candidate_nodes = self._get_all_process_nodes(session, target_process_key)
                    else:
                        logger.info("특정 질의 - 벡터 검색으로 관련 노드 필터링")
                        candidate_nodes = self._search_nodes_in_process(session, target_process_key, avg_vec)
                else:
                    logger.info("프로세스 미확정 - 전체 노드 벡터 검색")
                    candidate_nodes = self._search_all_nodes(session, avg_vec)
                
                logger.info(f"최종 후보 노드 수: {len(candidate_nodes)}")
                for i, node in enumerate(candidate_nodes[:5]):  # 상위 5개만 로깅
                    logger.info(f"  {i+1}. {node.type}: {node.name} ({node.key})")
                
                if len(candidate_nodes) > 5:
                    logger.info(f"  ... 외 {len(candidate_nodes) - 5}개 노드")
                
                logger.info("=== 3단계: 2단계 벡터 검색 완료 ===\n")
                return candidate_nodes
                
            except Exception as e:
                logger.error(f"2단계 벡터 검색 중 오류: {e}")
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
                return []

    def _get_all_process_nodes(self, session, process_key: str) -> List[GraphNode]:
        """프로세스 내 모든 노드 조회"""
        logger.debug(f"프로세스 내 전체 노드 조회: {process_key}")
        
        result = session.run("""
            MATCH (p:Process {key: $process_key})-[:CONTAINS]->(n)
            RETURN n.key as key, n.name as name, labels(n)[0] as type,
                   coalesce(n.searchText_i18n, n.searchText) as search_text,
                   properties(n) as properties
        """, process_key=process_key)
        
        nodes = []
        for record in result:
            nodes.append(GraphNode(
                key=record["key"],
                name=record["name"],
                type=record["type"],
                search_text=record["search_text"] or "",
                properties=record["properties"]
            ))
        
        logger.debug(f"프로세스 내 노드 조회 완료: {len(nodes)}개")
        return nodes

    def _search_nodes_in_process(self, session, process_key: str, 
                                query_vec: np.ndarray, top_k: int = 20) -> List[GraphNode]:
        """특정 프로세스 내 노드 벡터 검색"""
        logger.debug(f"프로세스 내 노드 벡터 검색: {process_key}")
        
        # 프로세스 내 노드 키 먼저 조회
        result = session.run("""
            MATCH (p:Process {key: $process_key})-[:CONTAINS]->(n)
            RETURN collect(n.key) as node_keys
        """, process_key=process_key)
        
        process_node_keys = result.single()["node_keys"]
        logger.debug(f"프로세스 내 노드 수: {len(process_node_keys)}")
        
        # 해당 노드들에 대해서만 벡터 검색
        return self._vector_search_with_filter(session, query_vec, process_node_keys, top_k)

    def _search_all_nodes(self, session, query_vec: np.ndarray, top_k: int = 20) -> List[GraphNode]:
        """전체 노드 벡터 검색"""
        logger.debug("전체 노드 벡터 검색 실행")
        return self._vector_search_with_filter(session, query_vec, None, top_k)

    def _vector_search_with_filter(self, session, query_vec: np.ndarray, 
                                  node_filter: Optional[List[str]] = None, 
                                  top_k: int = 20) -> List[GraphNode]:
        """필터링된 벡터 검색"""
        logger.debug(f"벡터 검색 실행 - 필터: {len(node_filter) if node_filter else 0}개, top_k: {top_k}")
        
        # 기존 벡터 검색 로직을 재사용하되 필터 적용
        unified_index = "idx_searchable_textEmbedding"
        
        if self._index_exists(session, unified_index):
            logger.debug(f"통합 인덱스 사용: {unified_index}")
            nodes = self._ann_search_index(session, unified_index, query_vec, k=max(50, top_k), ef=600)
        else:
            logger.debug("개별 인덱스 폴백 실행")
            nodes = []
            for idx in ["idx_activity_textEmbedding", "idx_gateway_textEmbedding", "idx_event_textEmbedding"]:
                if self._index_exists(session, idx):
                    logger.debug(f"인덱스 검색: {idx}")
                    nodes.extend(self._ann_search_index(session, idx, query_vec, k=max(50, top_k), ef=600))
        
        logger.debug(f"벡터 검색 원본 결과: {len(nodes)}개")
        
        # 필터 적용
        if node_filter:
            original_count = len(nodes)
            nodes = [node for node in nodes if node.key in node_filter]
            logger.debug(f"필터링 후 노드 수: {len(nodes)}개 (원본: {original_count}개)")
        
        return nodes[:top_k]

    def determine_bpmn_strategy(self, query: str, query_type: QueryType, 
                               candidate_nodes: List[GraphNode]) -> Dict[str, Any]:
        """BPMN 전문 용어를 사용한 컨텍스트 수집 전략 결정"""
        logger.info("=== 4단계: BPMN 전략 결정 시작 ===")
        logger.info(f"전략 결정 입력 - 질의타입: {query_type.value}, 후보노드수: {len(candidate_nodes)}")
        
        candidate_info = "\n".join([
            f"- {node.type}: {node.name} ({node.key})" 
            for node in candidate_nodes[:5]
        ])
        
        strategy_prompt = f"""
다음 BPMN 질의를 분석하여 어떤 프로세스 정보를 수집해야 하는지 결정해주세요.

**사용자 질의:** {query}
**질의 타입:** {query_type.value}
**관련 노드들:**
{candidate_info}

**BPMN 분석 전략 옵션:**
1. **연결 범위**: 
   - "direct_connections": 직접 연결된 활동들만
   - "gateway_branches": 게이트웨이를 통한 분기 경로까지
   - "full_process_chain": 전체 프로세스 체인
   
2. **관계 타입**: ["NEXT", "ROUTE", "RACI"] 중 필요한 것들

3. **세부 분석**:
   - "gateway_conditions": 게이트웨이 분기 조건 상세 분석
   - "path_tracing": 특정 경로 추적
   - "role_mapping": RACI 역할 매핑
   - "process_overview": 전체 프로세스 구조 파악

**JSON 형식으로 답변:**
{{
    "connection_scope": "direct_connections|gateway_branches|full_process_chain",
    "relationship_types": ["NEXT", "ROUTE", "RACI"] 중 필요한 것들,
    "gateway_conditions": true/false,
    "path_tracing": true/false,
    "role_mapping": true/false,
    "process_overview": true/false,
    "reasoning": "전략 선택 이유"
}}
"""
        
        try:
            logger.debug("OpenAI API 호출 - BPMN 전략 결정")
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            strategy_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM 전략 응답: {strategy_text}")
            
            if "```json" in strategy_text:
                strategy_text = strategy_text.split("```json")[1].split("```")[0].strip()
            elif "```" in strategy_text:
                strategy_text = strategy_text.split("```")[1].split("```")[0].strip()
            
            strategy = json.loads(strategy_text)
            logger.info(f"전략 결정 완료: {strategy.get('reasoning', '')}")
            logger.info(f"  - 연결 범위: {strategy.get('connection_scope')}")
            logger.info(f"  - 관계 타입: {strategy.get('relationship_types')}")
            logger.info(f"  - 세부 분석: 게이트웨이({strategy.get('gateway_conditions')}), 경로추적({strategy.get('path_tracing')}), 역할매핑({strategy.get('role_mapping')})")
            logger.info("=== 4단계: BPMN 전략 결정 완료 ===\n")
            return strategy
            
        except Exception as e:
            logger.error(f"전략 결정 오류: {e}")
            logger.warning("기본 전략 사용")
            strategy = self._get_default_bpmn_strategy(query_type)
            logger.info(f"기본 전략 적용: {strategy.get('reasoning')}")
            return strategy

    def _get_default_bpmn_strategy(self, query_type: QueryType) -> Dict[str, Any]:
        """기본 BPMN 분석 전략"""
        if query_type == QueryType.FLOW_OVERVIEW:
            return {
                "connection_scope": "full_process_chain",
                "relationship_types": ["NEXT", "ROUTE", "RACI"],
                "gateway_conditions": True,
                "path_tracing": False,
                "role_mapping": True,
                "process_overview": True,
                "reasoning": "전체 프로세스 흐름 파악을 위한 기본 전략"
            }
        elif query_type == QueryType.PATH_FINDING:
            return {
                "connection_scope": "gateway_branches",
                "relationship_types": ["NEXT", "ROUTE"],
                "gateway_conditions": True,
                "path_tracing": True,
                "role_mapping": False,
                "process_overview": False,
                "reasoning": "경로 찾기를 위한 기본 전략"
            }
        else:
            return {
                "connection_scope": "direct_connections",
                "relationship_types": ["NEXT", "ROUTE", "RACI"],
                "gateway_conditions": False,
                "path_tracing": False,
                "role_mapping": False,
                "process_overview": False,
                "reasoning": "일반 질의를 위한 기본 전략"
            }

    def collect_context(self, candidate_nodes: List[GraphNode], 
                       strategy: Dict[str, Any],
                       image_requested: bool = False) -> Dict[str, Any]:
        """향상된 컨텍스트 수집"""
        logger.info("=== 5단계: 컨텍스트 수집 시작 ===")
        logger.info(f"컨텍스트 수집 - 노드수: {len(candidate_nodes)}, 이미지요청: {image_requested}")
        
        with self.driver.session() as session:
            context = {
                "strategy": strategy,
                "nodes_info": {},
                "connections": {},
                "gateway_routes": [],
                "raci_mappings": {},
                "paths": [],
                "process_images": {}
            }
            
            if not candidate_nodes:
                logger.warning("후보 노드가 없어 빈 컨텍스트 반환")
                return context
            
            node_keys = [node.key for node in candidate_nodes]
            
            try:
                # BPMN 전략에 따른 연결 범위 설정
                scope = strategy.get("connection_scope", "direct_connections")
                if scope == "direct_connections":
                    max_depth = 1
                elif scope == "gateway_branches":
                    max_depth = 2
                else:  # full_process_chain
                    max_depth = 3
                
                logger.info(f"5-1. 기본 연결 정보 수집 - 최대깊이: {max_depth}")
                
                rel_types = strategy.get("relationship_types", ["NEXT", "ROUTE"])
                rel_pattern = "|".join(rel_types)
                logger.debug(f"관계 타입 패턴: {rel_pattern}")
                
                # 기본 연결 정보 수집
                basic_query = f"""
                    MATCH (n)
                    WHERE n.key IN $node_keys
                    
                    OPTIONAL MATCH path = (n)-[:{rel_pattern}*1..{max_depth}]-(connected)
                    
                    RETURN n.key as node_key, n.name as node_name, 
                           labels(n)[0] as node_type, properties(n) as node_props,
                           [node in nodes(path) | {{
                               key: node.key, 
                               name: node.name, 
                               type: labels(node)[0]
                           }}] as connected_nodes
                """
                
                result = session.run(basic_query, node_keys=node_keys)
                
                connection_count = 0
                for record in result:
                    node_key = record["node_key"]
                    context["nodes_info"][node_key] = {
                        "name": record["node_name"],
                        "type": record["node_type"],
                        "properties": record["node_props"]
                    }
                    
                    if record["connected_nodes"]:
                        context["connections"][node_key] = record["connected_nodes"]
                        connection_count += len(record["connected_nodes"])
                
                logger.info(f"기본 연결 정보 수집 완료 - 연결 수: {connection_count}")
                
                # 게이트웨이 분기 조건 분석
                if strategy.get("gateway_conditions", False):
                    logger.info("5-2. 게이트웨이 분기 조건 분석")
                    gateway_query = """
                        MATCH (gw:Gateway)-[route:ROUTE]->(target)
                        WHERE gw.key IN $node_keys OR target.key IN $node_keys
                        RETURN gw.key as gateway_key, gw.name as gateway_name,
                               target.key as target_key, target.name as target_name,
                               route.condition as condition, route.method as method, 
                               route.resp as resp, route.label as label
                    """
                    
                    gateway_result = session.run(gateway_query, node_keys=node_keys)
                    gateway_count = 0
                    for record in gateway_result:
                        context["gateway_routes"].append({
                            "gateway_key": record["gateway_key"],
                            "gateway_name": record["gateway_name"],
                            "target_key": record["target_key"],
                            "target_name": record["target_name"],
                            "condition": record["condition"],
                            "method": record["method"],
                            "resp": record["resp"],
                            "label": record["label"]
                        })
                        gateway_count += 1
                    
                    logger.info(f"게이트웨이 분기 조건 수집 완료 - {gateway_count}개")
                
                # RACI 역할 매핑
                if strategy.get("role_mapping", False):
                    logger.info("5-3. RACI 역할 매핑 수집")
                    raci_query = """
                        MATCH (role:Role)-[r:RACI]->(n)
                        WHERE n.key IN $node_keys
                        RETURN n.key as node_key, role.name as role_name, r.type as raci_type
                    """
                    
                    raci_result = session.run(raci_query, node_keys=node_keys)
                    raci_count = 0
                    for record in raci_result:
                        node_key = record["node_key"]
                        if node_key not in context["raci_mappings"]:
                            context["raci_mappings"][node_key] = []
                        
                        context["raci_mappings"][node_key].append({
                            "role": record["role_name"],
                            "type": record["raci_type"]
                        })
                        raci_count += 1
                    
                    logger.info(f"RACI 매핑 수집 완료 - {raci_count}개")
                
                # 경로 추적
                if strategy.get("path_tracing", False):
                    logger.info("5-4. 경로 추적 수집")
                    path_query = f"""
                        MATCH (start), (end)
                        WHERE start.key IN $node_keys AND end.key IN $node_keys
                            AND start <> end
                        MATCH path = shortestPath((start)-[:{rel_pattern}*1..{max_depth}]-(end))
                        RETURN [node in nodes(path) | {{
                            key: node.key, name: node.name, type: labels(node)[0]
                        }}] as path_nodes,
                        start.key as start_key, end.key as end_key
                        LIMIT 10
                    """
                    
                    path_result = session.run(path_query, node_keys=node_keys)
                    path_count = 0
                    for record in path_result:
                        context["paths"].append({
                            "start_key": record["start_key"],
                            "end_key": record["end_key"],
                            "nodes": record["path_nodes"]
                        })
                        path_count += 1
                    
                    logger.info(f"경로 추적 수집 완료 - {path_count}개")
                
                # 이미지 요청 시 프로세스 이미지 정보 수집
                if image_requested:
                    logger.info("5-5. 프로세스 이미지 정보 수집")
                    process_manager = ProcessManager(self.redis_client, self.session_id)
                    current_process = process_manager.get_current_process()
                    
                    if current_process:
                        process_image_query = """
                            MATCH (p:Process {key: $process_key})
                            RETURN p.bpmn_bucket as bucket, p.bpmn_pdf_key as pdf_key
                        """
                        
                        image_result = session.run(process_image_query, process_key=current_process["key"])
                        record = image_result.single()
                        
                        if record and record["bucket"] and record["pdf_key"]:
                            context["process_images"] = {
                                "bucket": record["bucket"],
                                "pdf_key": record["pdf_key"]
                            }
                            logger.info(f"프로세스 이미지 정보 발견 - bucket: {record['bucket']}, key: {record['pdf_key']}")
                        else:
                            logger.warning("프로세스 이미지 정보를 찾을 수 없습니다")
                    else:
                        logger.warning("현재 프로세스가 설정되지 않아 이미지 정보를 수집할 수 없습니다")
                
                logger.info("=== 5단계: 컨텍스트 수집 완료 ===")
                logger.info(f"수집된 컨텍스트 요약:")
                logger.info(f"  - 노드 정보: {len(context['nodes_info'])}개")
                logger.info(f"  - 연결 정보: {len(context['connections'])}개")
                logger.info(f"  - 게이트웨이 라우트: {len(context['gateway_routes'])}개")
                logger.info(f"  - RACI 매핑: {len(context['raci_mappings'])}개")
                logger.info(f"  - 경로: {len(context['paths'])}개")
                logger.info(f"  - 이미지 정보: {'있음' if context['process_images'] else '없음'}\n")
                
                return context
                
            except Exception as e:
                logger.error(f"컨텍스트 수집 중 오류: {e}")
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
                return context

    def get_query_embeddings(self, query: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """질의 임베딩 생성 (한국어, 영어, 평균)"""
        logger.info("=== 2단계: 질의 임베딩 생성 시작 ===")
        logger.info(f"임베딩 생성 대상: {query}")
        
        try:
            # 한국어 임베딩
            logger.debug("한국어 임베딩 생성")
            ko_vec = np.array(openai.embeddings.create(
                input=query, model="text-embedding-3-large"
            ).data[0].embedding)
            logger.debug(f"한국어 임베딩 완료 - 차원: {ko_vec.shape}")
        except Exception as e:
            logger.error(f"한국어 임베딩 생성 오류: {e}")
            ko_vec = np.zeros(3072, dtype=np.float32)

        try:
            # 영어 임베딩
            if self._contains_hangul(query):
                logger.debug("한국어 감지 - 영어 번역 후 임베딩")
                en = self.translate_query_to_english(query)
                logger.debug(f"번역 결과: {en}")
            else:
                logger.debug("영어 질의 - 직접 임베딩")
                en = query
                
            en_vec = np.array(openai.embeddings.create(
                input=en, model="text-embedding-3-large"
            ).data[0].embedding)
            logger.debug(f"영어 임베딩 완료 - 차원: {en_vec.shape}")
        except Exception as e:
            logger.error(f"영어 임베딩 생성 오류: {e}")
            en_vec = np.zeros(3072, dtype=np.float32)

        # 평균 앙상블
        if ko_vec.any() and en_vec.any():
            avg = ((ko_vec + en_vec) / 2.0).astype(np.float32)
            logger.info("한국어+영어 평균 앙상블 임베딩 생성")
        else:
            avg = ko_vec if ko_vec.any() else en_vec
            logger.warning("단일 언어 임베딩 사용")

        logger.info(f"최종 임베딩 차원: {avg.shape}")
        logger.info("=== 2단계: 질의 임베딩 생성 완료 ===\n")
        return avg, ko_vec, en_vec

    def translate_query_to_english(self, korean_query: str) -> str:
        """한국어 질의를 영어로 번역"""
        logger.debug(f"번역 요청: {korean_query}")
        
        translation_prompt = f"""
다음 한국어 BPMN 프로세스 질의를 영어로 번역해주세요.

한국어: {korean_query}

영어로만 답변하세요:
"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": translation_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"번역 완료: {result}")
            return result
        except Exception as e:
            logger.error(f"번역 오류: {e}")
            return korean_query

    def generate_answer(self, query: str, query_context: QueryContext,
                       candidate_nodes: List[GraphNode], 
                       graph_context: Dict[str, Any]) -> str:
        """향상된 답변 생성"""
        logger.info("=== 6단계: 답변 생성 시작 ===")
        logger.info(f"답변 생성 - 질의타입: {query_context.query_type.value}")
        
        try:
            # 컨텍스트 포맷팅
            logger.debug("컨텍스트 텍스트 포맷팅")
            context_text = self._format_context(candidate_nodes, graph_context, query_context.query_type)
            context_length = len(context_text)
            logger.debug(f"포맷된 컨텍스트 길이: {context_length}자")
            
            # 질의 타입별 시스템 프롬프트
            if query_context.query_type == QueryType.FLOW_OVERVIEW:
                system_prompt = """당신은 BPMN 프로세스 전문가입니다. 

**중요한 제약사항:**
1. 반드시 주어진 그래프 정보에만 기반하여 답변하세요
2. 그래프에 없는 정보는 절대 추가하지 마세요
3. Activity, Gateway, Event의 연결 관계와 흐름을 명확히 설명하세요
4. RACI 역할 정보가 있다면 포함하세요

주어진 그래프 정보를 바탕으로 비즈니스 프로세스의 전체적인 흐름을 설명해주세요."""
                
            elif query_context.query_type == QueryType.PATH_FINDING:
                system_prompt = """당신은 BPMN 경로 분석 전문가입니다.
주어진 그래프 정보를 바탕으로 특정 노드 간의 경로를 찾아 설명해주세요.
게이트웨이의 분기 조건과 라우팅 규칙도 자세히 설명해주세요."""
                
            else:
                system_prompt = """당신은 BPMN 분석 전문가입니다.
주어진 그래프 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요."""
            
            prompt = f"""
**사용자 질문:** {query}

**관련 그래프 정보:**
{context_text}

**답변 요구사항:**
- 위 그래프 정보에만 기반하여 답변
- 그래프 구조와 관계를 명확히 설명
- 한국어로 답변
"""
            
            logger.debug("OpenAI API 호출 - 답변 생성")
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content
            logger.info(f"답변 생성 완료 - 길이: {len(answer)}자")
            logger.info("=== 6단계: 답변 생성 완료 ===\n")
            
            return answer
            
        except Exception as e:
            error_msg = f"답변 생성 중 오류가 발생했습니다: {e}"
            logger.error(error_msg)
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return error_msg

    def _format_context(self, candidate_nodes: List[GraphNode], 
                       graph_context: Dict[str, Any], 
                       query_type: QueryType) -> str:
        """향상된 컨텍스트 포맷팅"""
        logger.debug("컨텍스트 포맷팅 시작")
        
        context_parts = []
        
        # 노드 정보
        context_parts.append("=== 관련 노드들 ===")
        for node in candidate_nodes:
            context_parts.append(f"- {node.type}: {node.name} ({node.key})")
        
        # 연결 정보
        if graph_context.get("connections"):
            context_parts.append("\n=== 프로세스 연결 구조 ===")
            for node_key, connections in graph_context["connections"].items():
                if connections:
                    node_info = graph_context["nodes_info"].get(node_key, {})
                    context_parts.append(f"{node_info.get('name', node_key)}와 연결된 노드들:")
                    for conn in connections:
                        if conn.get('name'):
                            context_parts.append(f"  - {conn['type']}: {conn['name']}")
        
        # 게이트웨이 분기 조건
        if graph_context.get("gateway_routes"):
            context_parts.append("\n=== 게이트웨이 분기 조건 ===")
            for route in graph_context["gateway_routes"]:
                conditions = []
                if route.get("condition"): conditions.append(f"조건: {route['condition']}")
                if route.get("method"): conditions.append(f"방법: {route['method']}")
                condition_text = ", ".join(conditions) if conditions else ""
                context_parts.append(f"  {route['gateway_name']} → {route['target_name']} ({condition_text})")
        
        # RACI 매핑
        if graph_context.get("raci_mappings"):
            context_parts.append("\n=== 역할 및 책임 (RACI) ===")
            for node_key, raci_list in graph_context["raci_mappings"].items():
                if raci_list:
                    node_info = graph_context["nodes_info"].get(node_key, {})
                    context_parts.append(f"{node_info.get('name', node_key)}:")
                    for raci in raci_list:
                        context_parts.append(f"  - {raci['role']}: {raci['type']}")
        
        # 경로 정보
        if graph_context.get("paths"):
            context_parts.append("\n=== 식별된 경로들 ===")
            for path in graph_context["paths"]:
                path_names = " → ".join([node.get('name', '') for node in path.get('nodes', [])])
                if path_names:
                    context_parts.append(f"  {path_names}")
        
        formatted_context = "\n".join(context_parts)
        logger.debug(f"컨텍스트 포맷팅 완료 - {len(context_parts)}개 섹션")
        
        return formatted_context

    def process_image_request(self, graph_context: Dict[str, Any]) -> Optional[str]:
        """이미지 요청 처리"""
        logger.info("=== 7단계: 이미지 요청 처리 시작 ===")
        
        process_images = graph_context.get("process_images", {})
        
        if not process_images.get("bucket") or not process_images.get("pdf_key"):
            logger.warning("프로세스 이미지 정보가 없습니다")
            logger.info("=== 7단계: 이미지 요청 처리 완료 (이미지 없음) ===\n")
            return None
        
        try:
            bucket_name = process_images['bucket']
            pdf_key = process_images['pdf_key']

            logger.info(f"MinIO 이미지 경로 생성 - bucket: {bucket_name}, key: {pdf_key}")
            
            CFG_FILE = "./config/bpmn.yaml"
            cfg = load_cfg(CFG_FILE)
            datalake_cfg = cfg["datalake"]

            client = MinIOClient(
                endpoint_url=datalake_cfg['endpoint'],
                access_key=datalake_cfg['access_key'],
                secret_key=datalake_cfg['secret_key']
            )

            image_path = client.get_pdf_path(bucket_name, pdf_key)

            if image_path:
                logger.info(f"이미지 경로 생성 성공: {image_path}")
            else:
                logger.error("이미지 경로 생성 실패")
                
            logger.info("=== 7단계: 이미지 요청 처리 완료 ===\n")
            return image_path
            
        except Exception as e:
            logger.error(f"이미지 요청 처리 중 오류: {e}")
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return None

    def handle_user_selection(self, selection: str) -> bool:
        """사용자 선택 처리"""
        logger.info(f"사용자 선택 처리: {selection}")
        
        process_manager = ProcessManager(self.redis_client, self.session_id)
        candidates = process_manager.get_process_candidates()
        
        logger.debug(f"후보 프로세스 수: {len(candidates)}")
        
        try:
            # 번호 선택 처리
            if selection.isdigit():
                idx = int(selection) - 1
                logger.debug(f"번호 선택: {idx + 1}")
                
                if 0 <= idx < len(candidates):
                    selected = candidates[idx]
                    process_manager.confirm_process(selected["key"], selected["name"])
                    logger.info(f"프로세스 선택 완료: {selected['name']}")
                    return True
                else:
                    logger.warning(f"잘못된 번호 선택: {idx + 1} (범위: 1-{len(candidates)})")
            
            # 이름으로 선택 처리
            for candidate in candidates:
                if selection.lower() in candidate["name"].lower():
                    process_manager.confirm_process(candidate["key"], candidate["name"])
                    logger.info(f"이름으로 프로세스 선택 완료: {candidate['name']}")
                    return True
            
            logger.warning(f"일치하는 프로세스를 찾을 수 없습니다: {selection}")
            return False
                    
        except Exception as e:
            logger.error(f"사용자 선택 처리 오류: {e}")
            return False

    def _index_exists(self, session, name: str) -> bool:
        rec = session.run("SHOW INDEXES YIELD name WHERE name=$n RETURN name", n=name, **self._ensure_db()).peek()
        return rec is not None

    def _ann_search_index(self, session, index_name: str, qvec: np.ndarray, k: int = 50, ef: int = 600) -> List[GraphNode]:
        logger.debug(f"ANN 검색 실행: {index_name}, k={k}")
        
        cypher4 = """
        CALL db.index.vector.queryNodes($index, $k, $vec, {ef: $ef})
        YIELD node, score
        RETURN labels(node)[0] AS type,
            node.key AS key,
            node.name AS name,
            coalesce(node.searchText_i18n, node.searchText) AS search_text,
            score,
            properties(node) AS props
        ORDER BY score DESC
        """
        params = {"index": index_name, "k": k, "vec": qvec.tolist(), "ef": ef, **self._ensure_db()}

        try:
            rows = session.run(cypher4, **params)
        except Neo4jError as e:
            logger.debug("4-parameter 검색 실패, 3-parameter로 폴백")
            cypher3 = """
            CALL db.index.vector.queryNodes($index, $k, $vec)
            YIELD node, score
            RETURN labels(node)[0] AS type,
                node.key AS key,
                node.name AS name,
                coalesce(node.searchText_i18n, node.searchText) AS search_text,
                score,
                properties(node) AS props
            ORDER BY score DESC
            """
            params.pop("ef", None)
            rows = session.run(cypher3, **params)

        nodes = []
        for r in rows:
            if r["key"] is None:
                continue
            nodes.append(
                GraphNode(
                    key=r["key"],
                    name=r["name"],
                    type=r["type"],
                    search_text=r["search_text"] or "",
                    embedding=None,
                    properties=r["props"],
                )
            )
        
        logger.debug(f"ANN 검색 결과: {len(nodes)}개 노드")
        return nodes

    def query(self, user_query: str) -> QueryResult:
        """메인 질의 처리 함수"""
        logger.info("=" * 60)
        logger.info("BPMN RAG 질의 처리 시작")
        logger.info("=" * 60)
        logger.info(f"사용자 질의: {user_query}")
        logger.info(f"세션 ID: {self.session_id}")
        
        try:
            # 1. 질의 맥락 분석
            query_context = self.analyze_query_context(user_query)
            
            # 사용자 상호작용이 필요한 경우
            if query_context.requires_user_interaction:
                logger.info("사용자 상호작용 필요 - 처리 중단")
                result = QueryResult(
                    query_type=query_context.query_type,
                    candidate_nodes=[],
                    graph_context={},
                    answer="",
                    image_path="",
                    requires_user_interaction=True,
                    interaction_message=query_context.interaction_message
                )
                logger.info("QueryResult 생성 완료 (상호작용)")
                logger.info("=" * 60)
                return result
            
            # 프로세스 선택 처리
            if query_context.process_action == "select_from_candidates":
                logger.info("프로세스 선택 처리 중")
                if self.handle_user_selection(user_query):
                    logger.info("선택 완료 - 재귀 호출")
                    return self.query("현재 프로세스에 대해 설명해주세요.")
                else:
                    logger.warning("잘못된 선택")
                    result = QueryResult(
                        query_type=QueryType.GENERAL,
                        candidate_nodes=[],
                        graph_context={},
                        answer="올바른 선택을 해주세요. 번호나 프로세스 이름을 입력해주세요.",
                        requires_user_interaction=True,
                        interaction_message="올바른 선택을 해주세요."
                    )
                    logger.info("=" * 60)
                    return result
            
            # 2. 질의 임베딩
            avg_vec, ko_vec, en_vec = self.get_query_embeddings(user_query)
            
            # 3. 2단계 벡터 검색
            candidate_nodes = self.two_stage_vector_search(query_context, avg_vec)
            
            if not candidate_nodes:
                logger.warning("후보 노드가 없습니다")
                result = QueryResult(
                    query_type=query_context.query_type,
                    candidate_nodes=[],
                    graph_context={},
                    answer="관련된 정보를 찾을 수 없습니다. 프로세스명을 다시 확인해주세요."
                )
                logger.info("=" * 60)
                return result
            
            # 4. BPMN 전략 결정
            strategy = self.determine_bpmn_strategy(user_query, query_context.query_type, candidate_nodes)
            
            # 5. 향상된 컨텍스트 수집
            graph_context = self.collect_context(candidate_nodes, strategy, query_context.image_requested)
            
            # 6. 답변 생성
            answer = self.generate_answer(user_query, query_context, candidate_nodes, graph_context)
            
            # 7. 이미지 처리
            image_path = None
            if query_context.image_requested:
                image_path = self.process_image_request(graph_context)
            
            # 8. 최종 결과 생성
            logger.info("=== 8단계: 최종 QueryResult 생성 ===")
            result = QueryResult(
                query_type=query_context.query_type,
                candidate_nodes=candidate_nodes,
                graph_context=graph_context,
                answer=answer,
                image_path=image_path
            )
            
            logger.info("QueryResult 생성 완료")
            logger.info(f"  - 질의 타입: {result.query_type.value}")
            logger.info(f"  - 후보 노드 수: {len(result.candidate_nodes)}")
            logger.info(f"  - 답변 길이: {len(result.answer)}자")
            logger.info(f"  - 이미지 경로: {'있음' if result.image_path else '없음'}")
            logger.info(f"  - 결과 타입: {type(result)}")
            
            logger.info("=" * 60)
            logger.info("BPMN RAG 질의 처리 완료")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            error_msg = f"질의 처리 중 오류가 발생했습니다: {e}"
            logger.error(error_msg)
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            
            result = QueryResult(
                query_type=QueryType.GENERAL,
                candidate_nodes=[],
                graph_context={},
                answer=error_msg
            )
            
            logger.error(f"오류 QueryResult 생성: {type(result)}")
            logger.info("=" * 60)
            return result