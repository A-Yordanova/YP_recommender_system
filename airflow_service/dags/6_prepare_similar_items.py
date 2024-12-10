import os
import logging

import pandas as pd
import numpy as np
import joblib
import pendulum
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from sqlalchemy import MetaData, Table, Column, String, Integer, BigInteger, Float, Boolean, DateTime, inspect, UniqueConstraint

from notifications import send_telegram_success_message, send_telegram_failure_message
from s3_functions import get_s3_session, read_parquet_from_s3, read_pkl_from_s3
from database_functions import extract_tables, get_sqlalchemy_type, create_dynamic_table, insert_dataframe_to_table


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

# DAG prepare_similar_items_recommendations parameters
@dag(
    schedule="00 23 * * *",  # Daily, at 23:00
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG prepare_similar_items_recommendations function
def prepare_similar_items_recommendations():
    logger.info(f"Start PREPARE_SIMILAR_ITEMS process.")

    # EXTRACT
    @task()
    def extract(**kwargs):
        """
        Establishes a connection to the database and
        downloads required tables from database and 
        trained encoders and ALS model from S3 to 
        temporary folder.

        Returns:
            extracted_data: dict of paths to data files.
        """
        logger.info(f"Start PREPARE_SIMILAR_ITEMS.extract subprocess.")
        
        try:
            db_conn = PostgresHook("postgresql_db").get_conn()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        # Load files
        files = {
            "recsys_items": "/tmp/recsys_items.csv",
            "recsys_events": "/tmp/recsys_events.csv",
        }
        extracted_data = extract_tables(db_conn, files)
        
        try:
            s3 = get_s3_session()
            
            # Load user_encoder
            user_encoder = read_pkl_from_s3(S3_BUCKET, "mle_final/models/user_encoder.pkl")
            user_encoder_path = "/tmp/user_encoder.pkl"
            joblib.dump(user_encoder, user_encoder_path)
            extracted_data["user_encoder"] = user_encoder_path
            
            # Load item_encoder
            item_encoder = read_pkl_from_s3(S3_BUCKET, "mle_final/models/item_encoder.pkl")
            item_encoder_path = "/tmp/item_encoder.pkl"
            joblib.dump(item_encoder, item_encoder_path)
            extracted_data["item_encoder"] = item_encoder_path
    
            # Load ALS-model
            als_model = read_pkl_from_s3(S3_BUCKET, "mle_final/models/als_model.pkl")
            als_model_path = "/tmp/als_model.pkl"
            joblib.dump(als_model, als_model_path)
            extracted_data["als_model"] = als_model_path
            
            logger.info(f"Encoders and ALS model have been successfully downloaded.")
        except Exception as e:
            logger.error(f"Failed to download trained encoders and ALS model from S3: {e}")

        # Two tables in files + 3 objects
        if len(extracted_data) < (len(files) + 3):
            logger.error("Extracted data is incomplete. Some tables failed to download.")
        
        logger.info(f"Finish PREPARE_SIMILAR_ITEMS.extract subprocess.")
        return extracted_data

    # CREATE RECOMMENDATIONS
    @task()
    def get_similar_items_recommendations(extracted_data):
        """
        Creates a table with N similar items for every 
        unique item_id in the items table. N is an 
        integer number according to 'n_recommendations' 
        setting in Airflow, if fails to load the setting
        then sets N=5.

        Parameters:
            extracted_data: dict of paths to files 
            containing 'items' and 'events' DataFrames,
            trained user and item encoders, and trained
            ALS model.

        Returns:
            created_recommendations: dict of paths to files 
            containing 'similar_items' DataFrame.
        """

        logger.info(f"Start PREPARE_SIMILAR_ITEMS.get_similar_items_recommendations subprocess.")
        
        # Load Airflow variables:
        # The number of similar_items for for each item_id.
        n_similar_items = int(Variable.get("n_similar_items", default_var=5))
        
        items = pd.read_csv(extracted_data["recsys_items"])
        events = pd.read_csv(extracted_data["recsys_events"])
        user_encoder = joblib.load(extracted_data["user_encoder"])
        item_encoder = joblib.load(extracted_data["item_encoder"])
        als_model = joblib.load(extracted_data["als_model"])

        # Encode items and events
        events["user_id_enc"] = user_encoder.transform(events["user_id"])
        items["item_id_enc"] = item_encoder.transform(items["item_id"])
        events["item_id_enc"] = item_encoder.transform(events["item_id"])

        # ======== SIMILAR ITEMS ========
        def get_similar_items(model, item_id, item_encoder, n_similar=5):
            item_id_enc = item_encoder.transform([item_id])[0]
            similar_items = model.similar_items(item_id_enc, N=n_similar+1)
            
            similar_item_ids_enc = similar_items[0]
            similarity_scores = similar_items[1]
            
            similar_item_ids = item_encoder.inverse_transform(similar_item_ids_enc)
            
            result = [(item_id, similar_item_id, score) 
                      for similar_item_id, score 
                      in zip(similar_item_ids, similarity_scores) 
                      if similar_item_id != item_id]
            
            return result
            
        def get_i2i_recommendations_for_items(items_df, model, item_encoder, n=5):
            unique_items = items["item_id"].unique().tolist()
            all_similar_items = []
            
            # For each unique item in the items df find N similar items
            for item_id in unique_items:
                item_similar_items = get_similar_items(model, item_id, item_encoder, n_similar=n)
                item_similar_items_flat = [(item_id, similar_item_id, score) for item_id, similar_item_id, score in item_similar_items]
                similar_items_df = pd.DataFrame(item_similar_items_flat, columns=["item_id", "similar_item_id", "score"])
                all_similar_items.append(similar_items_df)
            recommendations = pd.concat(all_similar_items, ignore_index=True)
            return recommendations
        
        # Create i2i recommendations for all items
        similar_items = get_i2i_recommendations_for_items(
            items, 
            als_model,
            item_encoder,
            n=n_similar_items
        )

        created_recommendations = {
            "similar_items": similar_items_path
        }

        logger.info(f"Finish PREPARE_SIMILAR_ITEMS.get_similar_items_recommendations subprocess.")
        return created_recommendations

    # LOAD
    @task()
    def load(created_recommendations):
        """
        Creates table in the database and loads prepared
        'similar_items' Dataframe to the table.

        Parameters:
            created_recommendations: dict of paths to 
            'similar_items' files.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start PREPARE_SIMILAR_ITEMS.load subprocess.")
        
        similar_items = pd.read_parquet(created_recommendations["similar_items"])
        
        try:
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        similar_items = similar_items.reset_index(drop=False)
        
        try:
            create_dynamic_table(
                db_connection=db_conn,
                df=similar_items,
                index_col="index",
                table_name="recsys_similar_items"
            )
            logger.info(f"Table recsys_similar_items have been created.")
        except Exception as e:
            logger.error(f"Table recsys_similar_items creation failed: {e}")
        
        try:
            insert_dataframe_to_table(
                db_connection=db_conn,
                df=similar_items,
                table_name="recsys_similar_items",
                if_exists="replace"
            )
            logger.info(f"Table recsys_similar_items have been inserted.")
        except Exception as e:
            logger.error(f"Inserting table recsys_similar_itemsrecsys_personalized_recommendations failed: {e}")
            
        logger.info(f"Finish PREPARE_SIMILAR_ITEMS.load subprocess.")


    extracted_data = extract()
    created_recommendations = get_similar_items_recommendations(extracted_data)
    load(created_recommendations)
    logger.info(f"Finish PREPARE_SIMILAR_ITEMS process.")

# Run the DAG
prepare_similar_items_recommendations()