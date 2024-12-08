FROM apache/airflow:2.7.3-python3.10

WORKDIR /opt/airflow/

USER root

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean

USER airflow

COPY requirements_airflow.txt ./tmp/requirements_airflow.txt
COPY .env /opt/airflow/.env

RUN pip install -U pip
RUN pip install --no-cache-dir -r ./tmp/requirements_airflow.txt