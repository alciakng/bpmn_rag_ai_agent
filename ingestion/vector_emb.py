from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from openai import OpenAI
from typing import List, Tuple
import streamlit as st
import numpy as np

# ── Secrets
NEO4J_URI  = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASS = st.secrets["NEO4J_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ── Models / Dim
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072
TRANSLATE_MODEL = "gpt-4o-mini"  # 한→영/영→한 번역용

# ── Neo4j / OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── Schema
LABELS   = ["Process", "Activity", "Gateway", "Event"]
TEXT_PROP = "searchText"            # 기존 영문 텍스트
I18N_PROP = "searchText_i18n"       # EN+KO 합본 텍스트
VEC_PROP  = "textEmbedding"         # 벡터 저장 프로퍼티

# =========================
# Helpers
# =========================
def translate_to_korean(text: str) -> str:
    """영문 → 한글 번역 (간결, 도메인 용어 보존)"""
    resp = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Translate to Korean naturally. Keep domain terms (BPMN, XOR, RACI, lane, etc.) as-is if needed. Return Korean only."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content.strip()

def to_i18n_text(en: str, ko: str) -> str:
    return f"EN: {en}\nKO: {ko}"

def embed_batch(texts: List[str]) -> List[List[float]]:
    """OpenAI 임베딩 배치 호출"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# =========================
# Cypher I/O
# =========================
def fetch_nodes_to_embed(tx, label: str, batch_size: int, skip: int) -> List[Tuple[int, str]]:
    """
    임베딩이 필요한 노드만 선택:
    - searchText 존재
    - (searchText_i18n이 없거나) OR (textEmbedding 없거나) OR (모델/차원이 다름)
    """
    q = f"""
    MATCH (n:{label})
    WHERE n.{TEXT_PROP} IS NOT NULL AND (
        n.{I18N_PROP} IS NULL OR
        n.{VEC_PROP}  IS NULL OR
        coalesce(n.embedding_model,'') <> $model OR
        coalesce(n.embedding_dim, 0)  <> $dim
    )
    RETURN id(n) AS id, n.{TEXT_PROP} AS en
    SKIP $skip LIMIT $limit
    """
    return [(r["id"], r["en"]) for r in tx.run(q, skip=skip, limit=batch_size, model=EMBED_MODEL, dim=EMBED_DIM)]

def write_i18n_and_embeddings(tx, rows: List[dict]):
    """
    rows: [{id, i18n, vec}]
    """
    q = f"""
    UNWIND $rows AS row
    MATCH (n) WHERE id(n)=row.id
    SET n.{I18N_PROP}     = row.i18n,
        n.{VEC_PROP}      = row.vec,
        n.embedding_model = $model,
        n.embedding_dim   = $dim,
        n.embedding_ts    = datetime()
    """
    tx.run(q, rows=rows, model=EMBED_MODEL, dim=EMBED_DIM).consume()

def ensure_vector_index(tx, label: str, index_name: str):
    """벡터 인덱스가 없으면 생성"""
    try:
        exists = tx.run(
            "SHOW INDEXES YIELD name WHERE name=$name RETURN name", name=index_name
        ).peek()
        if exists:
            print(f"[vector-index] exists: {index_name}")
            return {"created": False, "name": index_name}
        q = f"""
        CREATE VECTOR INDEX {index_name}
        FOR (n:{label}) ON (n.{VEC_PROP})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine',
            `vector.hnsw.m`: 16,
            `vector.hnsw.ef_construction`: 100
          }}
        }};
        """
        tx.run(q, dim=EMBED_DIM).consume()
        print(f"[vector-index] created: {index_name} (label={label}, dim={EMBED_DIM})")
        return {"created": True, "name": index_name}
    except Neo4jError as e:
        if e.code in {
            "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
            "Neo.ClientError.General.IndexAlreadyExists",
        }:
            print(f"[vector-index] race: already exists -> {index_name}")
            return {"created": False, "name": index_name}
        print(f"[vector-index][neo4j-error] name={index_name} code={e.code} msg={e.message}")
        raise
    except Exception as e:
        print(f"[vector-index][client-error] name={index_name} err={e}")
        raise

# =========================
# Main pipeline
# =========================
def embed_all_missing(batch_size: int = 64):
    with driver.session(database="neo4j") as s:
        for label in LABELS:
            skip = 0
            while True:
                rows = s.execute_read(fetch_nodes_to_embed, label, batch_size, skip)
                if not rows:
                    break

                ids, en_texts = zip(*rows)

                # ── 번역 (개별 호출; 문장 수 많으면 여기 배치 최적화 가능)
                ko_texts = []
                for en in en_texts:
                    try:
                        ko_texts.append(translate_to_korean(en))
                    except Exception as te:
                        # 번역 실패 시 영어만으로라도 임베딩(리콜은 다소 낮아짐)
                        print(f"[translate-fail] use EN only: {str(te)} :: {en[:120]}")
                        ko_texts.append("")  # 빈 한국어

                i18n_texts = [to_i18n_text(en, ko) if ko else f"EN: {en}" for en, ko in zip(en_texts, ko_texts)]

                # ── 임베딩
                vecs = embed_batch(i18n_texts)

                # ── 배치 쓰기
                payload = [{"id": nid, "i18n": i18n, "vec": vec}
                           for nid, i18n, vec in zip(ids, i18n_texts, vecs)]
                try:
                    s.execute_write(write_i18n_and_embeddings, payload)

                # ── 배치 실패 → 개별 폴백(어느 id가 실패했는지 확인)
                except Neo4jError as e:
                    print(f"[batch-write][neo4j] label={label} skip={skip} size={len(ids)} "
                          f"code={e.code} msg={e.message}")
                    for p in payload:
                        try:
                            s.execute_write(write_i18n_and_embeddings, [p])
                        except Neo4jError as e2:
                            print(f"[row-write-fail][neo4j] id={p['id']} code={e2.code} msg={e2.message} "
                                  f"text={p['i18n'][:120]}")
                        except Exception as e2:
                            print(f"[row-write-fail][client] id={p['id']} err={e2} text={p['i18n'][:120]}")

                except Exception as e:
                    print(f"[batch-write][client] label={label} skip={skip} err={e}")
                    for p in payload:
                        try:
                            s.execute_write(write_i18n_and_embeddings, [p])
                        except Exception as e2:
                            print(f"[row-write-fail][client] id={p['id']} err={e2} text={p['i18n'][:120]}")

                skip += batch_size

            # ── 라벨별 벡터 인덱스 보장
            s.execute_write(ensure_vector_index, label, f"idx_{label.lower()}_{VEC_PROP}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    try:
        embed_all_missing()
    finally:
        driver.close()
