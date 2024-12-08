FROM python:3.11-slim
COPY . ./fast_api_service
WORKDIR fast_api_service
RUN pip3 install -r requirements.txt
EXPOSE 8081
CMD uvicorn recsys_app.recsys_app:app --reload --port ${APP_PORT} --host 0.0.0.0