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
from rec_functions import get_available_items


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

# DAG prepare_default_recommendations parameters
@dag(
    schedule="30 22 * * *",  # Daily, at 22:30
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG prepare_default_recommendations function
def prepare_default_recommendations():
    logger.info(f"Start PREPARE_DEFAULT_RECOMMENDATIONS process.")

    # EXTRACT
    @task()
    def extract(**kwargs):
        """
        Establishes a connection to the database and
        downloads required tables to temporary folder.

        Returns:
            extracted_data: dict of paths to data files.
        """

        logger.info(f"Start PREPARE_DEFAULT_RECOMMENDATIONS.extract subprocess.")
        
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

        logger.info(f"Finish PREPARE_DEFAULT_RECOMMENDATIONS.extract subprocess.")
        return extracted_data

    # CREATE RECOMMENDATIONS
    @task()
    def get_default_recommendations(extracted_data):
        """
        Creates a table with top N popular items as default 
        recommendations; N is an integer number according 
        to 'num_default_recommendations' setting in Airflow,
        if fails to load the setting then sets N=100.

        Parameters:
            extracted_data: dict of paths to files 
            containing 'items' and 'events' DataFrames.

        Returns:
            created_recommendations: dict of paths to files 
            containing 'default_recommendations' DataFrame.
        """

        logger.info(f"Start PREPARE_DEFAULT_RECOMMENDATIONS.get_default_recommendations_for_cold_users subprocess.")
        
        items = pd.read_csv(extracted_data["recsys_items"])
        events = pd.read_csv(extracted_data["recsys_events"])

        num_recommendations = int(Variable.get("num_default_recommendations", default_var=100))
       
        # ======== DEFAULT RECOMMENDATIONS ========
        
        available_items = get_available_items(items)
        default_recommendations = events[
            ((events["rating"]>1)
             &(events["item_id"].isin(available_items)))
            ].groupby("item_id")["user_id"].count()
        
        default_recommendations = default_recommendations.sort_values(ascending=False).head(num_recommendations)
        default_recommendations = pd.DataFrame(
            {"item_id": default_recommendations.index,
             "count": default_recommendations.values}
        )
        default_recommendations = default_recommendations.sort_values(by="count", ascending=False)
                
        default_recommendations_path = "/tmp/default_recommendations.parquet"
        default_recommendations.to_parquet(default_recommendations_path)
        
        created_recommendations = {
            "default_recommendations": default_recommendations_path,
        }

        logger.info(f"Finish PREPARE_DEFAULT_RECOMMENDATIONS.get_default_recommendations_for_cold_users subprocess.")
        return created_recommendations

    # LOAD
    @task()
    def load(created_recommendations):
        """
        Creates table in the database and loads prepared
        'default_recommendations' Dataframe to the table.

        Parameters:
            created_recommendations: dict of paths to 
            'default_recommendations' files.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start PREPARE_DEFAULT_RECOMMENDATIONS.load subprocess.")
        
        default_recommendations = pd.read_parquet(created_recommendations["default_recommendations"])
        
        try:
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        default_recommendations = default_recommendations.reset_index(drop=False)
        
        try:
            create_dynamic_table(
                db_connection=db_conn, 
                df=default_recommendations, 
                index_col="index", 
                table_name="recsys_default_recommendations"
            )
            logger.info(f"Table recsys_default_recommendations have been created.")
        except Exception as e:
            logger.error(f"Table recsys_default_recommendations creation failed: {e}")

        try:
            insert_dataframe_to_table(
                db_connection=db_conn,
                df=default_recommendations,
                table_name="recsys_default_recommendations", 
                if_exists="replace"
            )
            logger.info(f"Table recsys_default_recommendations have been inserted.")
        except Exception as e:
            logger.error(f"Inserting table recsys_default_recommendations failed: {e}")
        
        logger.info(f"Finish PREPARE_DEFAULT_RECOMMENDATIONS.load subprocess.")
    
    extracted_data = extract()
    created_recommendations = get_default_recommendations(extracted_data)
    load(created_recommendations)
    logger.info(f"Finish PREPARE_DEFAULT_RECOMMENDATIONS process.")

# Run the DAG
prepare_default_recommendations()