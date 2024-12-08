import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

def create_postgresql_connection():
    load_dotenv()
    host = os.environ.get("DB_DESTINATION_HOST")
    port = os.environ.get("DB_DESTINATION_PORT")
    db = os.environ.get("DB_DESTINATION_NAME")
    username = os.environ.get("DB_DESTINATION_USER")
    password = os.environ.get("DB_DESTINATION_PASSWORD")
    
    postgresql_connection = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{db}")
    return postgresql_connection