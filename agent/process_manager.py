from pathlib import Path
from venv import logger

import sys
import numpy as np
import openai
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional
from dataclasses import dataclass
from agent.dto.process_state import ProcessState
from neo4j.exceptions import Neo4jError

class ProcessManager:
    def __init__(self, redis_client, session_id: str):
        self.redis = redis_client
        self.session_id = session_id
        self.ttl = 3600  # 1시간
        
    def _get_key(self, suffix: str) -> str:
        return f"session:{self.session_id}:{suffix}"

    def _index_exists(self, session, name: str) -> bool:
        rec = session.run("SHOW INDEXES YIELD name WHERE name=$n RETURN name", n=name, **self._ensure_db()).peek()
        return rec is not None
    
    def _ensure_db(self):
        return {"database": "neo4j"}
    
    def extract_process_name(self, query: str) -> Optional[str]:
        """질의에서 프로세스명 추출 (LLM 기반)"""
        extraction_prompt = f"""
다음 질의에서 BPMN 프로세스명을 추출해주세요.

질의: {query}

프로세스명 추출 규칙:
1. "~프로세스", "~과정", "~절차", "~워크플로우" 등의 패턴 찾기
2. 명시적으로 언급된 프로세스명만 추출
3. 프로세스명이 없으면 "NONE" 반환

출력 형식: 프로세스명만 출력 (예: "고객만족", "결재승인") 또는 "NONE"
"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            result = response.choices[0].message.content.strip()
            return None if result == "NONE" else result
        except Exception as e:
            print(f"프로세스명 추출 오류: {e}")
            return None
    
    def search_processes(self, process_name: str, neo4j_session, query_embedding: np.ndarray) -> List[Dict[str, str]]:
        """프로세스명의 임베딩을 기준으로 Process 노드 벡터 검색"""
        try:
            logger.debug(f"프로세스 벡터 검색: {process_name}")
            
            # Process 노드 전용 벡터 인덱스 사용
            process_index = "idx_process_textEmbedding"
            
            if not self._index_exists(neo4j_session, process_index):
                logger.warning(f"프로세스 인덱스 '{process_index}'가 존재하지 않습니다. 텍스트 검색으로 폴백합니다.")
                # 벡터 인덱스가 없으면 기존 텍스트 검색으로 폴백
                return self._search_processes_by_text(process_name, neo4j_session)
            
            # 벡터 검색 실행 (4-parameter 먼저 시도)
            cypher_query = """
                CALL db.index.vector.queryNodes($index, $k, $vec, {ef: $ef})
                YIELD node, score
                WHERE node:Process
                RETURN node.key as process_key, 
                        node.name as process_name,
                        score
                ORDER BY score DESC
            """
            
            params = {
                "index": process_index,
                "k": 10,  # 상위 10개 프로세스
                "vec": query_embedding.tolist(),
                "ef": 200  # 검색 정확도
            }
            
            try:
                result = neo4j_session.run(cypher_query, **params)
            except Neo4jError as e:
                logger.debug("4-parameter 벡터 검색 실패, 3-parameter로 폴백")
                # 3-parameter 폴백
                cypher_query = """
                    CALL db.index.vector.queryNodes($index, $k, $vec)
                    YIELD node, score
                    WHERE node:Process
                    RETURN node.key as process_key, 
                            node.name as process_name,
                            score
                    ORDER BY score DESC
                """
                params.pop("ef")
                result = neo4j_session.run(cypher_query, **params)
            
            processes = []
            for record in result:
                if record["process_key"]:  # null 값 제외
                    processes.append({
                        "key": record["process_key"],
                        "name": record["process_name"],
                        "score": record["score"]  # 유사도 점수 포함
                    })
            
            logger.info(f"벡터 검색 결과: {len(processes)}개 프로세스 발견")
            for i, proc in enumerate(processes[:3]):  # 상위 3개만 로깅
                logger.info(f"  {i+1}. {proc['name']} (점수: {proc['score']:.3f})")
            
            return processes
            
        except Exception as e:
            logger.error(f"프로세스 벡터 검색 오류: {e}")
            # 오류 시 텍스트 검색으로 폴백
            return self._search_processes_by_text(process_name, neo4j_session)
   
    def _search_processes_by_text(self, process_name: str, neo4j_session) -> List[Dict[str, str]]:
        """기존 텍스트 기반 프로세스 검색 (폴백용)"""
        try:
            result = neo4j_session.run("""
                MATCH (p:Process)
                WHERE p.name CONTAINS $process_name 
                    OR p.searchText CONTAINS $process_name
                RETURN p.key as process_key, p.name as process_name
                LIMIT 10
            """, process_name=process_name)
            
            processes = []
            for record in result:
                processes.append({
                    "key": record["process_key"],
                    "name": record["process_name"]
                })
            return processes
            
        except Exception as e:
            logger.error(f"텍스트 프로세스 검색 오류: {e}")
            return []
    
    def get_current_state(self) -> ProcessState:
        """현재 프로세스 상태 조회"""
        state = self.redis.get(self._get_key("state"))
        if state:
            return ProcessState(state.decode('utf-8'))
        return ProcessState.INITIAL
    
    def set_process_candidates(self, process_candidates: List[Dict[str, str]]):
        """프로세스 후보들 저장"""
        self.redis.setex(
            self._get_key("process_candidates"), 
            self.ttl,
            json.dumps(process_candidates)
        )
        self.redis.setex(self._get_key("state"), self.ttl, ProcessState.MULTIPLE_FOUND.value)
    
    def confirm_process(self, process_key: str, process_name: str):
        """프로세스 확정"""
        self.redis.setex(self._get_key("confirmed_process_key"), self.ttl, process_key)
        self.redis.setex(self._get_key("confirmed_process_name"), self.ttl, process_name)
        self.redis.setex(self._get_key("state"), self.ttl, ProcessState.CONFIRMED.value)
        # 후보 목록 제거
        self.redis.delete(self._get_key("process_candidates"))
    
    def get_current_process(self) -> Optional[Dict[str, str]]:
        """현재 확정된 프로세스 조회"""
        key = self.redis.get(self._get_key("confirmed_process_key"))
        name = self.redis.get(self._get_key("confirmed_process_name"))
        
        if key and name:
            return {
                "key": key.decode('utf-8'),
                "name": name.decode('utf-8')
            }
        return None
    
    def get_process_candidates(self) -> List[Dict[str, str]]:
        """프로세스 후보 목록 조회"""
        candidates = self.redis.get(self._get_key("process_candidates"))
        if candidates:
            return json.loads(candidates.decode('utf-8'))
        return []
    
    def handle_process_change(self, new_process_name: str) -> str:
        """프로세스 변경 처리"""
        current = self.get_current_process()
        if current:
            self.redis.setex(self._get_key("state"), self.ttl, ProcessState.CHANGING.value)
            return f"현재 '{current['name']}'를 분석 중입니다. '{new_process_name}' 프로세스로 변경하시겠습니까? (네/아니오)"
        return ""
    
    def reset_process(self):
        """프로세스 정보 초기화"""
        keys_to_delete = [
            self._get_key("confirmed_process_key"),
            self._get_key("confirmed_process_name"), 
            self._get_key("process_candidates"),
            self._get_key("state")
        ]
        self.redis.delete(*keys_to_delete)