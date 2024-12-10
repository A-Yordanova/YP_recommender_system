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
from rec_functions import get_available_items, get_user_history, get_eligible_users


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

# DAG prepare_personalized_recommendations parameters
@dag(
    schedule="45 22 * * *",  # Daily, at 22:45
    start_date=pendulum.datetime(2024, 11, 25, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG prepare_personalized_recommendations function
def prepare_personalized_recommendations():
    logger.info(f"Start PREPARE_PERSONALIZED_RECOMMENDATIONS process.")

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
        logger.info(f"Start PREPARE_PERSONALIZED_RECOMMENDATIONS.extract subprocess.")
        
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
        
        logger.info(f"Finish PREPARE_PERSONALIZED_RECOMMENDATIONS.extract subprocess.")
        return extracted_data

    # CREATE RECOMMENDATIONS
    @task()
    def get_personalized_recommendations_for_eligible_users(extracted_data):
        """
        Creates a table with N personalized recommendations 
        for all eligible users using ALS model; N is an 
        integer number according to 'n_recommendations' 
        setting in Airflow, if fails to load the setting
        then sets N=5; users become eligible for personalized
        recommendations if they have more or equal to 
        'pers_recommendations_threshold' unique items interactions;
        by default pers_recommendations_threshold = 3.

        Parameters:
            extracted_data: dict of paths to files 
            containing 'items' and 'events' DataFrames,
            trained user and item encoders, and trained
            ALS model.

        Returns:
            created_recommendations: dict of paths to files 
            containing 'personalized_recommendations' DataFrame.
        """

        logger.info(f"Start PREPARE_PERSONALIZED_RECOMMENDATIONS.get_personalized_recommendations_for_eligible_users subprocess.")

        # Load Airflow variables:
        # The number of unique items a user must interact with before becoming eligible for personalized recommendations.
        pers_threshold = int(Variable.get("pers_recommendations_threshold", default_var=3))
        # The number of personalized recommendations for each user.
        n_recommendations = int(Variable.get("n_pers_recommendations", default_var=5))
        
        items = pd.read_csv(extracted_data["recsys_items"])
        events = pd.read_csv(extracted_data["recsys_events"])
        user_encoder = joblib.load(extracted_data["user_encoder"])
        item_encoder = joblib.load(extracted_data["item_encoder"])
        als_model = joblib.load(extracted_data["als_model"])

        # Encode items and events
        events["user_id_enc"] = user_encoder.transform(events["user_id"])
        items["item_id_enc"] = item_encoder.transform(items["item_id"])
        events["item_id_enc"] = item_encoder.transform(events["item_id"])

        # Create sparse matrix for ALS model
        user_item_matrix = csr_matrix(
            (events["rating"],
             (events["user_id_enc"], events["item_id_enc"])
            ),
            dtype=np.uint8)

        # Create list of user who have enough interactions
        eligible_users = get_eligible_users(events, pers_threshold)
        
        # ======== PERSONALIZED RECOMMENDATIONS ========
        
        def get_personalized_recommendations(
            user_item_matrix, model, user_id, 
            user_encoder, item_encoder, items_df, n=5
        ):
            # Prepare list of available items
            available_items = get_available_items(items_df)

            # Encode user_ids
            user_id_enc = user_encoder.transform([user_id])[0]

            # Make personalized recommendations with ALS model
            recommendations = model.recommend(
                user_id_enc, 
                user_item_matrix[user_id_enc], 
                filter_already_liked_items=True,  # Exclude seen items
                N=n*10                            # Make 10 times more recommendatios in case we filter out a lot of duplicates
            )
            recommendations = pd.DataFrame({
                "user_id": user_id,
                "item_id_enc": recommendations[0], 
                "score": recommendations[1]
            })
            recommendations["item_id"] = item_encoder.inverse_transform(recommendations["item_id_enc"])
            recommendations = recommendations[recommendations["item_id"].isin(available_items)]
            return recommendations[["user_id", "item_id", "score"]].head(n)
        
        # Create personalized recommendations for all eligible users
        recommendations = []
        
        for user in eligible_users:
            user_recommendations = get_personalized_recommendations(
                user_item_matrix, 
                als_model, 
                user, 
                user_encoder, 
                item_encoder, 
                items, 
                n=n_recommendations
            )
        recommendations.append(user_recommendations)
        
        personalized_recommendations = pd.concat(recommendations, ignore_index=True)

        personalized_recommendations_path = "/tmp/personalized_recommendations.parquet"
        personalized_recommendations.to_parquet(personalized_recommendations_path)

        created_recommendations = {
            "personalized_recommendations": personalized_recommendations_path
        }
        
        logger.info(f"Finish PREPARE_PERSONALIZED_RECOMMENDATIONS.get_personalized_recommendations_for_eligible_users subprocess.")
        return created_recommendations

    # LOAD
    @task()
    def load(created_recommendations):
        """
        Creates table in the database and loads prepared
        'personalized_recommendations' Dataframe to the table.

        Parameters:
            created_recommendations: dict of paths to 
            'personalized_recommendations' files.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start PREPARE_PERSONALIZED_RECOMMENDATIONS.load subprocess.")
        
        personalized_recommendations = pd.read_parquet(created_recommendations["personalized_recommendations"])
        
        try:
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        personalized_recommendations = personalized_recommendations.reset_index(drop=False)

        try:
            create_dynamic_table(
                db_connection=db_conn,
                df=personalized_recommendations,
                index_col="index",
                table_name="recsys_personalized_recommendations"
            )
            logger.info(f"Table recsys_personalized_recommendations have been created.")
        except Exception as e:
            logger.error(f"Table recsys_personalized_recommendations creation failed: {e}")

        try:
            insert_dataframe_to_table(
                db_connection=db_conn,
                df=personalized_recommendations,
                table_name="recsys_personalized_recommendations",
                if_exists="replace"
            )
            logger.info(f"Table recsys_personalized_recommendations have been inserted.")
        except Exception as e:
            logger.error(f"Inserting table recsys_personalized_recommendations failed: {e}")
            
        logger.info(f"Finish PREPARE_PERSONALIZED_RECOMMENDATIONS.load subprocess.")

    extracted_data = extract()
    created_recommendations = get_personalized_recommendations_for_eligible_users(extracted_data)
    load(created_recommendations)
    logger.info(f"Finish PREPARE_PERSONALIZED_RECOMMENDATIONS process.")

# Run the DAG
prepare_personalized_recommendations()