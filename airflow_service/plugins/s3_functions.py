import os
from io import BytesIO
import joblib
from dotenv import load_dotenv
import boto3
import pyarrow.parquet as pq

load_dotenv("/opt/airflow/.env")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

def get_s3_session():
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET
        )

def read_parquet_from_s3(bucket, s3_key):
    s3 = get_s3_session()
    parquet_object = s3.get_object(Bucket=bucket, Key=s3_key)
    parquet_buffer = BytesIO(parquet_object["Body"].read())
    parquet_df = pq.read_table(parquet_buffer).to_pandas()
    return parquet_df

def read_pkl_from_s3(bucket, s3_key):
    s3 = get_s3_session()
    pkl_object = s3.get_object(Bucket=bucket, Key=s3_key)
    pkl_buffer = BytesIO(pkl_object["Body"].read())
    pkl_object = joblib.load(pkl_buffer)
    return pkl_object