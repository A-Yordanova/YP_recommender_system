import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
import pendulum
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
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
    schedule="45 23 25 * *",  # Monthly, 25th, at 23:45
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

def train_ranking_model_and_rank_recommendations():

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

        # Load personalized recommendations
        personalized_recommendations = pd.read_sql("SELECT * FROM recsys_personalized_recommendations", db_conn, index_col="index")
        personalized_recommendations_path = "/tmp/personalized_recommendations.parquet"
        personalized_recommendations.to_parquet(personalized_recommendations_path)

        # Load i2i recommendations
        i2i_recommendations = pd.read_sql("SELECT * FROM recsys_i2i_recommendations", db_conn, index_col="index")
        i2i_recommendations_path = "/tmp/i2i_recommendations.parquet"
        i2i_recommendations.to_parquet(i2i_recommendations_path)

        extracted_data = {
            "items": items_path,
            "events": events_path,
            "personalized_recommendations": personalized_recommendations_path, 
            "i2i_recommendations": i2i_recommendations_path
        }

        return extracted_data

    # PREPARE CANDIDATES FOR RANKING
    @task()
    def prepare_candidates(extracted_data):
        items = pd.read_parquet(extracted_data["items"])
        events = pd.read_parquet(extracted_data["events"])
        personalized_recommendations = pd.read_parquet(extracted_data["personalized_recommendations"])
        i2i_recommendations = pd.read_parquet(extracted_data["i2i_recommendations"])

        # Split data
        current_date = datetime.now()
        split_date = current_date - timedelta(days=7)
        split_idx = events["timestamp"] < split_date

        events_train = events[split_idx]
        events_test = events[~split_idx]

        # Prepare new features
        user_categories = pd.merge(events_train, items[["item_id", "category_id", "parent_category_id"]], on="item_id", how="left")
        category_count = user_categories.groupby(["user_id", "category_id"]).size().reset_index(name="count")
        parent_category_count = user_categories.groupby(["user_id", "parent_category_id"]).size().reset_index(name="count")
        
        user_favorite_category = category_count.loc[category_count.groupby("user_id")["count"].idxmax()]
        user_favorite_category = user_favorite_category[["user_id", "category_id"]].rename(columns={"category_id": "favorite_category_id"})
        
        user_favorite_parent_category = parent_category_count.loc[parent_category_count.groupby("user_id")["count"].idxmax()]
        user_favorite_parent_category = user_favorite_parent_category[["user_id", "parent_category_id"]]
        user_favorite_parent_category = user_favorite_parent_category.rename(columns={"parent_category_id": "favorite_parent_category_id"})
        
        user_favorite_categories = pd.merge(user_favorite_category, user_favorite_parent_category, on="user_id", how="outer")       

        # Prepare candidates
        candidates = pd.merge(
            personalized_recommendations[["user_id", "item_id", "score"]].rename(columns={"score": "pers_score"}),
            i2i_recommendations[["user_id", "item_id", "score"]].rename(columns={"score": "i2i_score"}),
            on=["user_id", "item_id"],
            how="outer"
        )
        
        candidates = candidates.merge(items[["item_id", "category_id", "parent_category_id"]], on="item_id", how="left")
        candidates = candidates.merge(user_favorite_categories, on="user_id", how="left")

        candidates_path = "/tmp/candidates.parquet"
        candidates.to_parquet(candidates_path)
        
        # Add target column:
        # — 1 for interacted item_id
        # — 0 for the rest
        
        events_train["target"] = 1
        candidates_for_training = candidates.merge(
            events_train[["user_id", "item_id", "target"]], 
            on=["user_id", "item_id"],
            how="left"
        )
        
        candidates_for_training["target"] = candidates_for_training["target"].fillna(0).astype("int")
        
        candidates_for_training_path = "/tmp/candidates_for_training.parquet"
        candidates_for_training.to_parquet(candidates_for_training_path)
        
        prepared_candidates = {"candidates": candidates_path, "candidates_for_training": candidates_for_training_path}
        
        return prepared_candidates

    # TRAIN RANKING MODEL AND RANK RECOMMENDATIONS
    @task()
    def retrain_ranking_model_and_rank_recommendations(prepared_candidates):
        candidates = pd.read_parquet(prepared_candidates["candidates"])
        candidates_for_training = pd.read_parquet(prepared_candidates["candidates_for_training"])
        
        features = ["pers_score", "i2i_score", "category_id", "parent_category_id", "favorite_category_id", "favorite_parent_category_id"]
        target = "target"
        
        train_data = Pool(
            data=candidates_for_training[features], 
            label=candidates_for_training[target]
        )
        
        ranking_model = CatBoostClassifier(
            learning_rate=0.05,
            depth=5,
            loss_function="Logloss",
            verbose=0,
            random_seed=42
        )
        
        ranking_model.fit(train_data)
        
        ranking_model_path = "/tmp/ranking_model.pkl"
        joblib.dump(ranking_model, ranking_model_path)

        s3 = get_s3_session()
        s3.upload_file(ranking_model_path, S3_BUCKET, "mle_final/models/ranking_model.pkl")

        inference_data = Pool(
            data=candidates[features]
        )
        
        predictions = ranking_model.predict_proba(inference_data)
        
        candidates["ranking_score"] = predictions[:, 1]
        
        candidates = candidates.sort_values(["user_id", "ranking_score"], ascending=[True, False])
        candidates["rank"] = candidates.groupby("user_id").cumcount() + 1
        
        max_recommendations_per_user = 20
        ranked_recommendations = candidates[candidates["rank"] <= max_recommendations_per_user]
        
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
        
        ranked_recommendations = ranked_recommendations.reset_index(drop=False)
        create_dynamic_table(df=ranked_recommendations, index_col="index", table_name="recsys_ranked_recommendations")

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
            
        insert_dataframe_to_table(ranked_recommendations, "recsys_ranked_recommendations")

    
    extracted_data = extract()
    prepared_candidates = prepare_candidates(extracted_data)
    retrain_ranking_model_and_rank_recommendations(prepared_candidates)

# Run the DAG
train_ranking_model_and_rank_recommendations()