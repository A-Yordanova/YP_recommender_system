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
    schedule="00 23 * * *",  # Daily, at 23:00
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

def prepare_similar_items_recommendations():

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

        s3 = get_s3_session()
        
        # Load user_encoder
        user_encoder = read_pkl_from_s3(S3_BUCKET, "mle_final/models/user_encoder.pkl")
        user_encoder_path = "/tmp/user_encoder.pkl"
        joblib.dump(user_encoder, user_encoder_path)
        
        # Load item_encoder
        item_encoder = read_pkl_from_s3(S3_BUCKET, "mle_final/models/item_encoder.pkl")
        item_encoder_path = "/tmp/item_encoder.pkl"
        joblib.dump(item_encoder, item_encoder_path)

        # Load ALS-model
        als_model = read_pkl_from_s3(S3_BUCKET, "mle_final/models/als_model.pkl")
        als_model_path = "/tmp/als_model.pkl"
        joblib.dump(als_model, als_model_path)

        extracted_data = {
            "items": items_path, "events": events_path, 
            "user_encoder": user_encoder_path, 
            "item_encoder": item_encoder_path, 
            "als_model": als_model_path
        }

        return extracted_data

    # CREATE RECOMMENDATIONS
    @task()
    def get_similar_items_recommendations(extracted_data):
        items = pd.read_parquet(extracted_data["items"])
        events = pd.read_parquet(extracted_data["events"])
        user_encoder = joblib.load(extracted_data["user_encoder"])
        item_encoder = joblib.load(extracted_data["item_encoder"])
        als_model = joblib.load(extracted_data["als_model"])

        # Encode items and events
        events["user_id_enc"] = user_encoder.transform(events["user_id"])
        items["item_id_enc"] = item_encoder.transform(items["item_id"])
        events["item_id_enc"] = item_encoder.transform(events["item_id"])

        # ======== SIMILAR ITEMS ========
        def get_similar_items(model, item_id, item_encoder, n_similar=3):
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
           
           for item_id in tqdm(unique_items):
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
            n=5
        )

        created_recommendations = {
            "similar_items": similar_items_path
        }
        
        return created_recommendations

    # LOAD
    @task()
    def load(created_recommendations):
        i2i_recommendations = pd.read_parquet(created_recommendations["i2i_recommendations"])
        
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

        i2i_recommendations = i2i_recommendations.reset_index(drop=False)
        create_dynamic_table(df=i2i_recommendations, index_col="index", table_name="recsys_i2i_recommendations")

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
            
        insert_dataframe_to_table(i2i_recommendations, "recsys_i2i_recommendations")
        

    extracted_data = extract()
    created_recommendations = get_i2i_recommendations_for_eligible_users(extracted_data)
    load(created_recommendations)

# Run the DAG
prepare_similar_items_recommendations()