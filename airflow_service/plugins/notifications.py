import os
from dotenv import load_dotenv
from airflow.providers.telegram.hooks.telegram import TelegramHook

load_dotenv("/opt/airflow/.env")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_failure_message(context):
    hook = TelegramHook(telegram_conn_id="airflow_telegram_notifications",
                        token=TELEGRAM_TOKEN,
                        chat_id=TELEGRAM_CHAT_ID)
    dag_info = context["dag"]
    run_id = context["run_id"]
    message = f"{dag_info} failed."
    hook.send_message({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": None
    })

def send_telegram_success_message(context):
    hook = TelegramHook(telegram_conn_id="airflow_telegram_notifications",
                        token=TELEGRAM_TOKEN,
                        chat_id=TELEGRAM_CHAT_ID)
    dag_info = context["dag"]
    run_id = context["run_id"]
    message = f"{dag_info} successfully finished."
    hook.send_message({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": None
    })