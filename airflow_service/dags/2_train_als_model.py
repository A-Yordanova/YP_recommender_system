import os
import logging

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


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv("/opt/airflow/.env")
    S3_BUCKET = os.getenv("S3_BUCKET_NAME")
    S3_KEY = os.getenv("S3_ACCESS_KEY_ID")
    S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY")
    MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    logger.info(f"Environment variables successfully loaded.")
except:
    logger.error(f"Environment variables loading failed.")

# DAG train_als_model parameters
@dag(
    schedule="15 22 25 * *",  # Monthly, 25th, at 22:15
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG train_als_model function
def train_als_model():
    logger.info(f"Start TRAIN_ALS_MODEL process.")

    # EXTRACT
    @task()
    def extract(**kwargs):
        """
        Establishes a connection to the database and
        downloads required tables to temporary folder.

        Returns:
            extracted_data: dict of paths to data files.
        """
        
        logger.info(f"Start TRAIN_ALS_MODEL.extract subprocess.")
        
        try:
            db_conn = PostgresHook("postgresql_db").get_conn()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        # Load tables
        files = {
            "recsys_items": "/tmp/recsys_items.csv",
            "recsys_events": "/tmp/recsys_events.csv",
        }
        extracted_data = extract_tables(db_conn, files)
                
        if len(extracted_data) < len(files):
            logger.error("Extracted data is incomplete. Some tables failed to download.")

        logger.info(f"Finish TRAIN_ALS_MODEL.extract subprocess.")
        return extracted_data

    # TRAIN ENCODERS
    @task()
    def encode(extracted_data):
        """
        Encodes user_ids and item_ids in 'events' 
        and 'items' Dataframes with LabelEncoder().
        Uploads trained encoders to S3.

        Parameters:
            extracted_data: dict of paths to files 
            containing 'items' and 'events' DataFrames.

        Returns:
            encoded_data: dict of paths to files 
            containing 'items' and 'events' DataFrames 
            with encoded ids.
        """

        logger.info(f"Start TRAIN_ALS_MODEL.encode subprocess.")
        
        items = pd.read_csv(extracted_data["recsys_items"])
        events = pd.read_csv(extracted_data["recsys_events"])

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

        try:
            s3 = get_s3_session()
            s3.upload_file(user_encoder_path, S3_BUCKET, "mle_final/models/user_encoder.pkl")
            s3.upload_file(item_encoder_path, S3_BUCKET, "mle_final/models/item_encoder.pkl")
            logger.info(f"Trained encoders successfully uploaded to S3.")
        except Exception as e:
                logger.error(f"Failed to upload trained encoders to S3: {e}")

        encoded_data = {"items": items_path, "events": events_path}

        logger.info(f"Finish TRAIN_ALS_MODEL.encode subprocess.")
        return encoded_data

    # TRAIN ALS-MODEL
    @task()
    def retrain_als_model(encoded_data):
        """
        Creates user_item matrix, trains ALS model
        on the matrix, uploads trained model to S3.

        Parameters:
            encoded_data: dict of paths to files 
            containing 'items' and 'events' DataFrames 
            with encoded ids.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start TRAIN_ALS_MODEL.retrain_als_model subprocess.")
        
        events = pd.read_parquet(encoded_data["events"])

        # Create sparse user-item matrix tfor ALS model
        user_item_matrix = csr_matrix(
            (events["rating"],
             (events["user_id_enc"], events["item_id_enc"])
            ),
            dtype=np.uint8
        )

        # Initialize AlternatingLeastSquares object
        als_model = AlternatingLeastSquares(factors=50, iterations=50, regularization=0.05, random_state=42)

        # Train the model
        als_model.fit(user_item_matrix)

        # Save model to temporary file
        als_model_path = "/tmp/als_model.pkl"
        joblib.dump(als_model, als_model_path)

        # Upload model file to S3
        try:
            s3 = get_s3_session()
            s3.upload_file(als_model_path, S3_BUCKET, "mle_final/models/als_model.pkl")
            logger.info(f"Trained ALS model successfully uploaded to S3.")
        except Exception as e:
            logger.error(f"Failed to upload trained ALS model to S3: {e}")
        
        logger.info(f"Finish TRAIN_ALS_MODEL.retrain_als_model subprocess.")

    extracted_data = extract()
    encoded_data = encode(extracted_data)
    retrain_als_model(encoded_data)
    logger.info(f"Finish TRAIN_ALS_MODEL process.")

# Run the DAG
train_als_model()