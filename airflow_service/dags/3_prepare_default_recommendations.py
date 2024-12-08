import os

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
from sqlalchemy import MetaData, Table, Column, String, Integer, BigInteger, Float, Boolean, DateTime, inspect, UniqueConstraint

from notifications import send_telegram_success_message, send_telegram_failure_message
from s3_functions import get_s3_session, read_parquet_from_s3, read_pkl_from_s3

load_dotenv("/opt/airflow/.env")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

@dag(
    schedule="30 22 * * *",  # Daily, at 22:30
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

def prepare_default_recommendations():

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

        extracted_data = {
            "items": items_path,
            "events": events_path
        }

        return extracted_data

    # CREATE RECOMMENDATIONS
    @task()
    def get_default_recommendations_for_cold_users(extracted_data):
        items = pd.read_parquet(extracted_data["items"])
        events = pd.read_parquet(extracted_data["events"])
       
        def get_available_items(items_df):
            available_items = items_df[items_df["is_available"]==1]["item_id"].tolist()
            return available_items
       
        # ======== DEFAULT RECOMMENDATIONS ========
        
        def get_default_recommendations(events_df, items_df):
            available_items = get_available_items(items_df)
            default_recommendations = events_df[
            ((events_df["rating"]>1)
             &(events_df["item_id"].isin(available_items)))
            ].groupby("item_id")["user_id"].count()
        
            default_recommendations = default_recommendations.sort_values(ascending=False).head(100)
            default_recommendations = pd.DataFrame({"item_id": default_recommendations.index,
                                                    "count": default_recommendations.values})
            default_recommendations = default_recommendations.sort_values(by="count", ascending=False)
            
            return default_recommendations
        
        # Create default recommendations for cold users
        default_recommendations = get_default_recommendations(events, items)
        
        default_recommendations_path = "/tmp/default_recommendations.parquet"
        default_recommendations.to_parquet(default_recommendations_path)
        
        created_recommendations = {
            "default_recommendations": default_recommendations_path,
        }
        
        return default_recommendations

    # LOAD
    @task()
    def load(created_recommendations):
        default_recommendations = pd.read_parquet(created_recommendations["default_recommendations"])
        
        type_mapping = {
            "int64": Integer,
            "float64": Float,
            "object": String,
            "bool": Boolean,
            "datetime64[ns]": DateTime
        }
        
        def get_sqlalchemy_type(pandas_type):
            """Get SQLAlchemy type based on pandas column type."""
            return type_mapping.get(str(pandas_type), String)
        
        def create_dynamic_table(df: pd.DataFrame, index_col, table_name: str):
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            metadata = MetaData()
            columns = []
            for column in df.columns:
                column_type = get_sqlalchemy_type(df[column].dtype)
                columns.append(Column(column, column_type)) 
            table = Table(table_name, metadata, *columns, UniqueConstraint(index_col, name=f"unique_{table_name}_id_constraint"))
            if not inspect(db_conn).has_table(table.name):
                metadata.create_all(db_conn)
        
        default_recommendations = default_recommendations.reset_index(drop=False)
        create_dynamic_table(df=default_recommendations, index_col="index", table_name="recsys_default_recommendations")


        def insert_dataframe_to_table(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
            """Insert a DataFrame into a PostgreSQL table."""
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            df.to_sql(
                name=table_name,
                con=db_conn,
                index=False,       # Exclude DataFrame index
                if_exists=if_exists,  # Options: 'fail', 'replace', 'append'
                chunksize=10000
            )
            
        insert_dataframe_to_table(default_recommendations, "recsys_default_recommendations")
        

    extracted_data = extract()
    created_recommendations = get_default_recommendations_for_cold_users(extracted_data)
    load(created_recommendations)

# Run the DAG
prepare_default_recommendations()