import os
import logging

import pandas as pd
import numpy as np
import psycopg2
import pendulum
from dotenv import load_dotenv
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import MetaData, Table, Column, String, Integer, BigInteger, Float, Boolean, DateTime, inspect, UniqueConstraint
from notifications import send_telegram_success_message, send_telegram_failure_message
from database_functions import extract_tables, get_sqlalchemy_type, create_dynamic_table, insert_dataframe_to_table


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# DAG prepare_datasets parameters
@dag(
    schedule="0 22 * * *",  # Daily, at 22:00
    start_date=pendulum.datetime(2024, 12, 4, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

# DAG prepare_datasets function
def prepare_datasets():
    logger.info(f"Start PREPARE_DATASETS process.")

    # EXTRACT
    @task()
    def extract(**kwargs):
        """
        Establishes a connection to the database and
        downloads required tables to temporary folder.

        Returns:
            extracted_data: dict of paths to data files.
        """

        logger.info(f"Start PREPARE_DATASETS.extract subprocess.")

        try:
            db_conn = PostgresHook("postgresql_db").get_conn()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")

        # Load tables
        files = {
            "category_tree": "/tmp/categories.csv",
            "item_properties": "/tmp/item_properties.csv",
            "events": "/tmp/events.csv"
        }
        
        extracted_data = extract_tables(db_conn, files)
                
        if len(extracted_data) < len(files):
            logger.error("Extracted data is incomplete. Some tables failed to download.")
        
        logger.info(f"Finish PREPARE_DATASETS.extract subprocess.")
        return extracted_data

    # TRANSFORM
    @task()
    def transform(extracted_data):
        """
        Transforms extracted tables, merges 'categories' 
        and 'item_properties' into 'items' DataFrame.

        Parameters:
            extracted_data: dict of paths to 'categories',
            'item_properties' and 'events' files.

        Returns:
            transformed_data: dict of paths to 'events' 
            and 'items' files.
        
        """

        logger.info(f"Start PREPARE_DATASETS.transform subprocess.")
        
        categories = pd.read_csv(extracted_data["category_tree"])
        item_properties = pd.read_csv(extracted_data["item_properties"])
        events = pd.read_csv(extracted_data["events"])

        # Transform categories df
        categories["parent_category_id"] = categories["parent_category_id"].fillna(-1).astype("int")

        # Transform item_properties df
        item_properties = item_properties.sort_values(by=["item_id", "property", "timestamp"], ascending=[True, True, False])
        item_properties = item_properties.drop_duplicates(subset=["item_id", "property"], keep="first")
        
        item_properties = item_properties.drop(columns=["timestamp"]).reset_index(drop=True)

        property_usage = item_properties["property"].value_counts()
        threshold_abs = int(item_properties["item_id"].nunique() * 0.2)
        frequent_properties = property_usage[property_usage >= threshold_abs].index
        filtered_item_properties = item_properties[item_properties["property"].isin(frequent_properties)]

        # Transform events df
        events["timestamp"] = pd.to_datetime(events["timestamp"], unit = "ms")
        events["timestamp"] = pd.to_datetime(events["timestamp"]).dt.floor("T")
        events["rating"] = events["event_type"].map({"view": 1, 
                                             "addtocart": 3, 
                                             "transaction": 5})
        events = events.drop(columns=["transaction_id", "event_type"])

        # Create items df
        items = filtered_item_properties.pivot_table(
            index="item_id", 
            columns="property", 
            values="property_value", 
            aggfunc="first"
        ).reset_index()
        
        items = items.rename(columns = {"categoryid" : "category_id", "available": "is_available"})
        items["is_available"] = items["is_available"].astype("int")
        items["category_id"] = items["category_id"].astype("int")

        # Merge items df and categories df
        items = items.merge(categories, on="category_id", how="left")
        items["parent_category_id"] = items["parent_category_id"].fillna(-1).astype("int")
        
        # Remove constant properties
        items_stats = pd.DataFrame({
            "column_name": items.columns,
            "missing_values": items.isna().sum().values,
            "unique_values_without_nan": [items[col].nunique(dropna=True) for col in items.columns]
        })
        columns_to_delete = items_stats[(
            (items_stats["missing_values"]==0)
            &(items_stats["unique_values_without_nan"]==1)
        )]["column_name"].tolist()
        items = items.drop(columns=columns_to_delete)
        items = items.fillna(-1)

        # Get valid item IDs from the `items` DataFrame
        valid_item_ids = set(items["item_id"])
        
        # Filter the `events` DataFrame to keep only rows with valid `item_id`
        events = events[events["item_id"].isin(valid_item_ids)]

        columns_to_str = items.select_dtypes(include="object").columns
        items[columns_to_str] = items[columns_to_str].astype("str")
        
        items_path = "/tmp/items.parquet"
        items.to_parquet(items_path)
        
        columns_to_str = events.select_dtypes(include="object").columns
        events[columns_to_str] = events[columns_to_str].astype("str")
        events_path = "/tmp/events.parquet"
        events.to_parquet(events_path)

        transformed_data = {"items": items_path, "events": events_path}

        logger.info(f"Finish PREPARE_DATASETS.transform subprocess.")
        return transformed_data

    # LOAD
    @task()
    def load(transformed_data):
        """
        Creates tables in the database and loads prepared
        'events' and 'items' Dataframes to the tables.

        Parameters:
            transformed_data: dict of paths to 'events' 
            and 'items' files.

        Returns:
            Doesn't return anything.
        """

        logger.info(f"Start PREPARE_DATASETS.load subprocess.")
        
        items = pd.read_parquet(transformed_data["items"])
        events = pd.read_parquet(transformed_data["events"])
        
        try:
            db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()
            logger.info(f"Database connection has been established.")
        except Exception as e:
            logger.error(f"Establishing connection failed: {e}")
        
        events = events.reset_index(drop=False)

        try:
            create_dynamic_table(
                db_connection=db_conn, 
                df=events, 
                index_col="id", 
                table_name="recsys_events"
            )
            create_dynamic_table(
                db_connection=db_conn, 
                df=items, 
                index_col="item_id", 
                table_name="recsys_items"
            )
            logger.info(f"Tables recsys_items and recsys_events have been created.")
        except Exception as e:
            logger.error(f"Tables recsys_items and recsys_events creation failed: {e}")

        try:
            insert_dataframe_to_table(
                db_connection=db_conn, 
                df=items, 
                table_name="recsys_items", 
                if_exists="replace"
            )
            insert_dataframe_to_table(
                db_connection=db_conn,
                df=events,
                table_name="recsys_events", 
                if_exists="replace"
            )
            logger.info(f"Tables recsys_items and recsys_events have been inserted.")
        except Exception as e:
            logger.error(f"Inserting tables recsys_items and recsys_events failed: {e}")
        
        logger.info(f"Finish PREPARE_DATASETS.load subprocess.")
        
    extracted_data = extract()
    transformed_data = transform(extracted_data)
    load(transformed_data)
    logger.info(f"Finish PREPARE_DATASETS process.")

# Run the DAG
prepare_datasets()