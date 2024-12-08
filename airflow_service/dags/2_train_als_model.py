import os

import pandas as pd
import numpy as np
import joblib
import pendulum
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

from notifications import send_telegram_success_message, send_telegram_failure_message
from s3_functions import get_s3_session

load_dotenv("/opt/airflow/.env")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

@dag(
    schedule="15 22 25 * *",  # Monthly, 25th, at 22:15
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

def train_als_model():

    # EXTRACT
    @task()
    def extract(**kwargs):
        db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()

        # Load Items
        items = pd.read_sql("SELECT * FROM recsys_items", db_conn)
        items_path = "/tmp/items.parquet"
        items.to_parquet(items_path)
        
        # Load Events
        events = pd.read_sql("SELECT * FROM recsys_events", db_conn, index_col="id")
        events_path = "/tmp/events.parquet"
        events.to_parquet(events_path)

        extracted_data = {"items": items_path, "events": events_path}

        return extracted_data

    # TRAIN ENCODERS
    @task()
    def encode(extracted_data):
        items = pd.read_parquet(extracted_data["items"])
        events = pd.read_parquet(extracted_data["events"])

        user_encoder = LabelEncoder()
        user_encoder.fit(events["user_id"])
        events["user_id_enc"] = user_encoder.transform(events["user_id"])

        item_encoder = LabelEncoder()
        item_encoder.fit(items["item_id"])
        items["item_id_enc"] = item_encoder.transform(items["item_id"])
        events["item_id_enc"] = item_encoder.transform(events["item_id"])
        
        events_path = "/tmp/events.parquet"
        events.to_parquet(events_path)

        items_path = "/tmp/items.parquet"
        items.to_parquet(items_path)
        
        user_encoder_path = "/tmp/user_encoder.pkl"
        joblib.dump(user_encoder, user_encoder_path)

        item_encoder_path = "/tmp/item_encoder.pkl"
        joblib.dump(item_encoder, item_encoder_path)

        s3 = get_s3_session()
        s3.upload_file(user_encoder_path, S3_BUCKET, "mle_final/models/user_encoder.pkl")
        s3.upload_file(item_encoder_path, S3_BUCKET, "mle_final/models/item_encoder.pkl")

        encoded_data = {"items": items_path, "events": events_path}
        
        return encoded_data

    # TRAIN ALS-MODEL
    @task()
    def retrain_als_model(encoded_data):
        events = pd.read_parquet(encoded_data["events"])
        user_item_matrix = csr_matrix(
            (events["rating"],
             (events["user_id_enc"], events["item_id_enc"])
            ),
            dtype=np.uint8
        )
        als_model = AlternatingLeastSquares(factors=50, iterations=50, regularization=0.05, random_state=42)
        als_model.fit(user_item_matrix)

        als_model_path = "/tmp/als_model.pkl"
        joblib.dump(als_model, als_model_path)

        s3 = get_s3_session()
        s3.upload_file(als_model_path, S3_BUCKET, "mle_final/models/als_model.pkl")

    extracted_data = extract()
    encoded_data = encode(extracted_data)
    retrain_als_model(encoded_data)

# Run the DAG
train_als_model()