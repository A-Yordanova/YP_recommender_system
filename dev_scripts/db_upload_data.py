import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import Integer, BigInteger, Float, String, UniqueConstraint, inspect
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Load environmental variables
dst_host = os.environ.get("DB_DESTINATION_HOST")
dst_port = os.environ.get("DB_DESTINATION_PORT")
dst_username = os.environ.get("DB_DESTINATION_USER")
dst_password = os.environ.get("DB_DESTINATION_PASSWORD")
dst_db = os.environ.get("DB_DESTINATION_NAME")

# Create connection to DB
dst_conn = create_engine(f"postgresql://{dst_username}:{dst_password}@{dst_host}:{dst_port}/{dst_db}")

# -------------------------------------------------------------------------------------------------------
# Load Categories
categories = pd.read_csv("data/category_tree.csv")

# Rename DataFrame columns
categories.rename(
    columns={
        "categoryid": "category_id",
        "parentid": "parent_category_id",
    },
    inplace=True
)

# Upload to PostgreSQL
metadata = MetaData()
table = Table(
    "category_tree", 
    metadata,
    Column("category_id", Integer, primary_key=True, autoincrement=False),
    Column("parent_category_id", Integer),
    UniqueConstraint("category_id", name="unique_category_id_constraint")
)

# Check if the table exists and create it if it doesn't
if not inspect(dst_conn).has_table(table.name):
    metadata.create_all(dst_conn)
    print(f"Table {table.name} created successfully.")
else:
    print(f"Table {table.name} already exists.")

categories.to_sql(
    name="category_tree",
    con=dst_conn,
    if_exists="append",  # Append data if the table exists
    index=False          # Don't include the DataFrame index
)
print("Data inserted successfully.")

sql = f"SELECT * FROM category_tree"
temp = pd.read_sql(sql, dst_conn).reset_index(drop=True)
print(temp.head(1))

# -------------------------------------------------------------------------------------------------------
# Load item_properties
item_properties = pd.read_csv("data/item_properties.csv")

# Rename DataFrame columns
item_properties.rename(
    columns={
        "itemid": "item_id",
        "timestamp": "timestamp", # unchanged
        "property": "property", # unchanged
        "value": "property_value"
    },
    inplace=True
)

# Upload to PostgreSQL
metadata = MetaData()
table = Table(
    "item_properties", 
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", BigInteger),
    Column("item_id", Integer),
    Column("property", String),
    Column("property_value", String),
    UniqueConstraint("id", name="unique_item_property_id_constraint")
)

# Check if the table exists and create it if it doesn't
if not inspect(dst_conn).has_table(table.name):
    metadata.create_all(dst_conn)
    print(f"Table {table.name} created successfully.")
else:
    print(f"Table {table.name} already exists.")

item_properties.to_sql(
    name="item_properties",
    con=dst_conn,
    if_exists="append",  # Append data if the table exists
    index=False          # Don't include the DataFrame index
)
print("Data inserted successfully.")

sql = f"SELECT * FROM item_properties"
temp = pd.read_sql(sql, dst_conn).reset_index(drop=True)
print(temp.head(1))

# -------------------------------------------------------------------------------------------------------
# Load Categories
events = pd.read_csv("data/events.csv")

# Rename DataFrame columns
events.rename(
    columns={
        "timestamp": "timestamp", # unchanged
        "visitorid": "user_id",
        "event": "event_type",
        "itemid": "item_id",
        "transactionid": "transaction_id"
    },
    inplace=True
)
# Upload to PostgreSQL
metadata = MetaData()
table = Table(
    "events", 
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", BigInteger),
    Column("user_id", Integer),
    Column("event_type", String),
    Column("item_id", Integer),
    Column("transaction_id", Float),
    UniqueConstraint("id", name="unique_event_id_constraint")
)

# Check if the table exists and create it if it doesn't
if not inspect(dst_conn).has_table(table.name):
    metadata.create_all(dst_conn)
    print(f"Table {table.name} created successfully.")
else:
    print(f"Table {table.name} already exists.")

events.to_sql(
    name="events",
    con=dst_conn,
    if_exists="append",  # Append data if the table exists
    index=False          # Don't include the DataFrame index
)
print("Data inserted successfully.")

sql = f"SELECT * FROM events"
temp = pd.read_sql(sql, dst_conn).reset_index(drop=True)
print(temp.head(1))