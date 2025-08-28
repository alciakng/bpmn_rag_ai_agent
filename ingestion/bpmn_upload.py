#%%
import sys
import os
from glob import glob

from neo4j import GraphDatabase

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

from helpers import load_cfg
from minio_utils import MinIOClient
import streamlit as st
###############################################
# Parameters & Arguments
###############################################
CFG_FILE = "./config/bpmn.yaml"

# ── Secrets
NEO4J_URI  = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASS = st.secrets["NEO4J_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
###############################################

###############################################
# Main
###############################################
def _upsert_process_pdf_key(neo4j_uri, neo4j_user, neo4j_password,
                            process_key, bucket, object_key):
    """업로드한 PDF를 Process 노드에 매핑 (이름으로 매칭)"""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    cypher = """
    MATCH (p:Process {key:$process_key})
    SET p.bpmn_bucket  = $bucket,
        p.bpmn_pdf_key = $object_key,
        p.updated_at   = datetime()
    RETURN p.key AS key, p.name AS name, p.bpmn_bucket AS bucket, p.bpmn_pdf_key AS pdf_key
    """
    with driver.session() as sess:
        rec = sess.run(cypher, process_key=process_key, bucket=bucket, object_key=object_key).single()
    driver.close()
    if not rec:
        print(f"[WARN] Process(key='{process_key}')를 찾지 못했습니다. (키 미설정)")
    else:
        print(f"[OK] {rec['name']} ← {rec['bucket']}/{rec['pdf_key']}")

#%%
def bpmn_upload(endpoint_url, access_key, secret_key):
    """ img&xml upload to s3(minio) bucket"""

    cfg = load_cfg(CFG_FILE)

    bpmn_cfg = cfg['bpmn']
    datalake_cfg = cfg["datalake"]

    client = MinIOClient(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key
    )

    client.create_bucket(datalake_cfg["bucket_name_1"])

    # Upload imgs
    all_imgs = glob(os.path.join(bpmn_cfg["img_path"], "*.pdf"))
    print(os.path.join(bpmn_cfg["img_path"], "*.pdf"))
    print(f"all_fps {all_imgs}")
    for img in all_imgs:
        print(f"Uploading {img}")
        base = os.path.basename(img)                            # "Customer Quotation (Sales Ex Works).pdf"
        process_name = os.path.splitext(base)[0]                # "Customer Quotation (Sales Ex Works)"
        object_key   = os.path.join(datalake_cfg["folder_name"], base).replace("\\", "/")

        client_minio = client.create_conn()
        client_minio.fput_object(
            bucket_name=datalake_cfg["bucket_name_1"],
            object_name=object_key,
            file_path=img,
        )

          # 3) 업로드 후, 해당 Process 노드에 키값 저장 (이름으로 매칭)
        _upsert_process_pdf_key(
            neo4j_uri      = NEO4J_URI,
            neo4j_user     = NEO4J_USER,
            neo4j_password = NEO4J_PASS,
            process_key    = process_name,
            bucket         = datalake_cfg["bucket_name_1"],
            object_key     = object_key
        )
    
###############################################
#%%
if __name__ == "__main__":
    cfg = load_cfg(CFG_FILE)

    datalake_cfg = cfg["datalake"]

    ENDPOINT_URL_LOCAL = datalake_cfg['endpoint']
    ACCESS_KEY_LOCAL = datalake_cfg['access_key']
    SECRET_KEY_LOCAL = datalake_cfg['secret_key']

    bpmn_upload(ENDPOINT_URL_LOCAL, ACCESS_KEY_LOCAL, SECRET_KEY_LOCAL)