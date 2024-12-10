import os
import logging
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

# DAG train_ranking_model_and_rank_recommendations parameters
@dag(
    schedule="45 23 25 * *",  # Monthly, 25th, at 23:45
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG train_ranking_model_and_rank_recommendations function
def train_ranking_model_and_rank_recommendations():

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
        
        logger.info(f"Start TRAIN_RANKING_MODEL.extract subprocess.")
        
        try:
            db_conn = PostgresHook("postgresql_db").get_conn()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        # Load files
        files = {
            "recsys_items": "/tmp/recsys_items.csv",
            "recsys_events": "/tmp/recsys_events.csv",
            "recsys_personalized_recommendations": "/tmp/recsys_personalized_recommendations.csv",
            "recsys_i2i_recommendations": "/tmp/recsys_i2i_recommendations.csv"
        }
        extracted_data = extract_tables(db_conn, files)
        
        if len(extracted_data) < (len(files) + 3):
            logger.error("Extracted data is incomplete. Some tables failed to download.")
        
        logger.info(f"Finish TRAIN_RANKING_MODEL.extract subprocess.")
        return extracted_data

    # PREPARE CANDIDATES FOR RANKING
    @task()
    def prepare_candidates(extracted_data):
        """
        Prepare the candidates table for ranking (CatBoost)
        model training. Split date for inference is a setting
        'ranking_model_inference_split_date' in Airflow.

        Parameters:
            extracted_data: dict of paths to data files.

        Returns:
            prepared_candidates: dict of paths to data files.
        """
        
        logger.info(f"Start TRAIN_RANKING_MODEL.prepare_candidates subprocess.")

        # Load Airflow variables:
        # The number of days to take for test set for the ranking model.
        inference_delta = int(Variable.get("ranking_model_inference_delta", default_var=7))
        
        items = pd.read_csv(extracted_data["recsys_items"])
        events = pd.read_csv(extracted_data["recsys_events"])
        personalized_recommendations = pd.read_csv(extracted_data["recsys_personalized_recommendations"])
        i2i_recommendations = pd.read_csv(extracted_data["recsys_i2i_recommendations"])

        # Split data based on set inference delta in days
        current_date = datetime.now()
        split_date = current_date - timedelta(days=inference_delta)
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

        # Prepare candidates frim two sources
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
        events_train["target"] = 1
        candidates_for_training = candidates.merge(
            events_train[["user_id", "item_id", "target"]], 
            on=["user_id", "item_id"],
            how="left"
        )
        
        # — 0 for the rest
        candidates_for_training["target"] = candidates_for_training["target"].fillna(0).astype("int")
        
        candidates_for_training_path = "/tmp/candidates_for_training.parquet"
        candidates_for_training.to_parquet(candidates_for_training_path)
        
        prepared_candidates = {"candidates": candidates_path, "candidates_for_training": candidates_for_training_path}

        logger.info(f"Finish TRAIN_RANKING_MODEL.prepare_candidates subprocess.")
        return prepared_candidates

    # TRAIN RANKING MODEL AND RANK RECOMMENDATIONS
    @task()
    def retrain_ranking_model_and_rank_recommendations(prepared_candidates):
        """
        Train ranking model on the candidates DataFrame 
        and apply in to rank combined recommendations 
        from two sources (personalized and i2i).

        Parameters:
            prepared_candidates: dict of paths to data files.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start TRAIN_RANKING_MODEL.retrain_ranking_model_and_rank_recommendations subprocess.")
        
        candidates = pd.read_parquet(prepared_candidates["candidates"])
        candidates_for_training = pd.read_parquet(prepared_candidates["candidates_for_training"])

        # Separate input features and target feature
        features = ["pers_score", "i2i_score", "category_id", "parent_category_id", "favorite_category_id", "favorite_parent_category_id"]
        target = "target"

        # Create Pool object with training data
        train_data = Pool(
            data=candidates_for_training[features], 
            label=candidates_for_training[target]
        )

        # Initialize CatBoostClassifier object
        ranking_model = CatBoostClassifier(
            learning_rate=0.05,
            depth=5,
            loss_function="Logloss",
            verbose=0,
            random_seed=42
        )

        # Train the model
        ranking_model.fit(train_data)

        # Save model in temporaty file
        ranking_model_path = "/tmp/ranking_model.pkl"
        joblib.dump(ranking_model, ranking_model_path)

        # Upload model file to S3
        try:
            s3 = get_s3_session()
            s3.upload_file(ranking_model_path, S3_BUCKET, "mle_final/models/ranking_model.pkl")
            logger.info(f"Trained ranking model successfully uploaded to S3.")
        except Exception as e:
            logger.error(f"Failed to upload trained ranking model to S3: {e}")

        # Create Pool object with all the data
        inference_data = Pool(
            data=candidates[features]
        )

        # Make rank predictions for all recommendations
        predictions = ranking_model.predict_proba(inference_data)
        
        candidates["ranking_score"] = predictions[:, 1]
        
        candidates = candidates.sort_values(["user_id", "ranking_score"], ascending=[True, False])
        candidates["rank"] = candidates.groupby("user_id").cumcount() + 1
        
        max_recommendations_per_user = 20
        ranked_recommendations = candidates[candidates["rank"] <= max_recommendations_per_user]

        # Saving ranked recommendations in the database
        try:
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        ranked_recommendations = ranked_recommendations.reset_index(drop=False)
        
        try:
            create_dynamic_table(
                db_connection=db_conn,
                df=ranked_recommendations,
                index_col="index",
                table_name="recsys_ranked_recommendations"
            )
            logger.info(f"Table recsys_ranked_recommendations have been created.")
        except Exception as e:
            logger.error(f"Table recsys_ranked_recommendations creation failed: {e}")
        
        try:
            insert_dataframe_to_table(
                db_connection=db_conn,
                df=ranked_recommendations,
                table_name="recsys_ranked_recommendations",
                if_exists="replace"
            )
            logger.info(f"Table recsys_ranked_recommendations have been inserted.")
        except Exception as e:
            logger.error(f"Inserting table recsys_ranked_recommendations failed: {e}")
            
        logger.info(f"Finish TRAIN_RANKING_MODEL.retrain_ranking_model_and_rank_recommendations subprocess.")

    
    extracted_data = extract()
    prepared_candidates = prepare_candidates(extracted_data)
    retrain_ranking_model_and_rank_recommendations(prepared_candidates)
    logger.info(f"Finish TRAIN_RANKING_MODEL process.")

# Run the DAG
train_ranking_model_and_rank_recommendations()