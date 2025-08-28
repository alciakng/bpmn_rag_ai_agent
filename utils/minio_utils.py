from datetime import timedelta
import os
from typing import Optional
from minio import Minio

class MinIOClient:
    def __init__(self, endpoint_url, access_key, secret_key):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key

    def create_conn(self):
        client = Minio(
            endpoint=self.endpoint_url,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
        )
        return client

    def create_bucket(self, bucket_name):
        client = self.create_conn()

        # Create bucket if not exist
        found = client.bucket_exists(bucket_name=bucket_name)
        if not found:
            client.make_bucket(bucket_name=bucket_name)
            print(f"Bucket {bucket_name} created successfully!")
        else:
            print(f"Bucket {bucket_name} already exists, skip creating!")
    
    def get_pdf_path(self, bucket_name: str, pdf_key: str) -> Optional[str]:

        client = self.create_conn()
        """MinIO에서 PDF 경로 생성"""
        try:
            # PDF 파일 존재 확인
            client.stat_object(bucket_name, pdf_key)
            # 프리사인된 URL 생성 (1시간 유효)
            return client.presigned_get_object(bucket_name, pdf_key, expires=timedelta(hours=1))
        except Exception as e:
            print(f"MinIO PDF 로드 오류: {e}")
            return None
            