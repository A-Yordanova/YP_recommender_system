import pandas as pd
import psycopg2
from sqlalchemy import MetaData, Table, Column, String, Integer, BigInteger, Float, Boolean, DateTime, inspect, UniqueConstraint


def extract_tables(db_connection, tables_to_extract):
    """
    Extract tables form DB and save table name and path to the file.
    """
    extracted_tables = {}
    for table_name, copy_path in tables_to_extract.items():
        with db_connection.cursor() as cur:
            with open(copy_path, "w") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)
        extracted_tables[table_name] = copy_path
 
    return extracted_tables

def get_sqlalchemy_type(pandas_type):
    """
    Get SQLAlchemy type based on pandas column type.
    """
    
    type_mapping = {
        "int64": Integer,
        "float64": Float,
        "object": String,
        "bool": Boolean,
        "datetime64[ns]": DateTime
    }
    return type_mapping.get(str(pandas_type), String)

def create_dynamic_table(db_connection, df: pd.DataFrame, index_col, table_name: str):
    """
    Create table in database based on DataFrame.
    """

    metadata = MetaData()
    columns = []
    for column in df.columns:
        column_type = get_sqlalchemy_type(df[column].dtype)
        columns.append(Column(column, column_type))
    table = Table(table_name, metadata, *columns, UniqueConstraint(index_col, name=f"unique_{table_name}_id_constraint"))
    if not inspect(db_conn).has_table(table.name):
        metadata.create_all(db_conn)

def insert_dataframe_to_table(db_connection, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    """
    Insert a DataFrame into a PostgreSQL table.
    """

    df.to_sql(
        name=table_name,
        con=db_conn,
        index=False,
        if_exists=if_exists,  # Options: 'fail', 'replace', 'append'
        chunksize=10000
    )