import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table

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

# Function for deleting a table with sqlalchemy
def delete_table(table_name, conn):
    metadata = MetaData()
    metadata.reflect(bind=conn)  # Reflect tables from the database
    if table_name in metadata.tables:
        table = metadata.tables[table_name]
        table.drop(conn)  # Drop the table
        print(f"Table '{table_name}' deleted successfully.")
    else:
        print(f"Table '{table_name}' does not exist in the database.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Delete a table from the database.")
parser.add_argument("table_name", type=str, help="Name of the table to delete.")
args = parser.parse_args()

# Assign table_name from arguments
table_name = args.table_name

# Delete the table
delete_table(table_name, dst_conn)