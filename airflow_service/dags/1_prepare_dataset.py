import os
import pandas as pd
import numpy as np
import pendulum
from dotenv import load_dotenv
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import MetaData, Table, Column, String, Integer, BigInteger, Float, Boolean, DateTime, inspect, UniqueConstraint
from notifications import send_telegram_success_message, send_telegram_failure_message

@dag(
    schedule="0 22 * * *",  # Daily, at 22:00
    start_date=pendulum.datetime(2024, 12, 4, tz="UTC"),
    catchup=False,
    tags=["recsys_ecommerce"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)

def prepare_datasets():

    # EXTRACT
    @task()
    def extract(**kwargs):
        db_conn = PostgresHook("postgresql_db").get_sqlalchemy_engine()

        # Load Categories
        categories = pd.read_sql("SELECT * FROM category_tree", db_conn).reset_index(drop=True)
        categories_path = "/tmp/categories.parquet"
        categories.to_parquet(categories_path)
        
        # Load Item Properties
        item_properties = pd.read_sql("SELECT * FROM item_properties", db_conn, index_col="id")
        item_properties_path = "/tmp/item_properties.parquet"
        item_properties.to_parquet(item_properties_path)
        
        # Load Events
        events = pd.read_sql("SELECT * FROM events", db_conn, index_col="id")
        events_path = "/tmp/events.parquet"
        events.to_parquet(events_path)

        extracted_data = {"categories": categories_path, "item_properties": item_properties_path, "events": events_path}

        return extracted_data

    # TRANSFORM
    @task()
    def transform(extracted_data):
        categories = pd.read_parquet(extracted_data["categories"])
        item_properties = pd.read_parquet(extracted_data["item_properties"])
        events = pd.read_parquet(extracted_data["events"])

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
        
        return transformed_data

    # LOAD
    @task()
    def load(transformed_data):
        items = pd.read_parquet(transformed_data["items"])
        events = pd.read_parquet(transformed_data["events"])
        
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
        
        events = events.reset_index(drop=False)
        create_dynamic_table(df=events, index_col="id", table_name="recsys_events")
        create_dynamic_table(df=items, index_col="item_id", table_name="recsys_items")

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
            
        insert_dataframe_to_table(items, "recsys_items")
        insert_dataframe_to_table(events, "recsys_events")

    extracted_data = extract()
    transformed_data = transform(extracted_data)
    load(transformed_data)

# Run the DAG
prepare_datasets()