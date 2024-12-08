# Recommender System for E-Commerce

## Project Goals
The Recommender System aims to provide personalized product recommendations to users based on their interaction history and preferences. The system utilizes collaborative filtering techniques, including implicit user feedback (views, adding to cart, purchases), to predict items users are likely to engage with. The goal is to optimize the recommendation process for different user segments, particularly for cold users (those with limited or no interaction history) and users with a richer history.

The goal is to improve the current set of evaluation metrics by enhancing recommendation relevance, novelty, and coverage. We aim to:
* Improve Precision and Recall for all user segments by improving the model's ability to predict relevant items.
* Maintain Novelty@5 to ensure the recommendations are diverse and not repetitive.
* Increase Coverage for Items to recommend a wider variety of products to users, beyond just the most popular ones.
* Refine Coverage for Users to ensure that all user segments, including those with sparse histories, are catered to effectively.

## Project Structure
Project consists of three parts:
1. **Modeling Experiments**. This part includes exploratory data analysis and model development, with all steps logged using MLFlow.
2. **Pipeline**. This part involves Airflow service for data loading and model re-training.
3. **Fast API Service**. This part covers the development of the microservice, its deployment in a Docker container, and monitoring using Prometheus and Grafana.

Raw data used in this project is stored in PostgresSQL database in the following tables: `category_tree`, `item_properties`, `events`.

## Project Setup Guide
Copy the repository with the project:
```bash
git clone https://github.com/A-Yordanova/mle-pr-final.git
```

Make sure all the commands are executed from the work directory:
```bash
cd mle-pr-final
```

Create and activate new virtual environment:
```
python3 -m venv .venv_mle_pr_final
source .venv_mle_pr_final/bin/activate
```

Install required Python libraries:
```bash
pip install -r requirements.txt
```

## 1. Modeling Experiments
Exploratory data analysis and modeling experiments are in the notebook `research.ipynb`. All experiments and models were logged with MLFlow.

**To launch Jupyter Server:**
```bash
sh dev_scripts/run_jupyter_server.sh
# Access: http://127.0.0.0:8888
```

**To launch MLFlow Server:**
```bash
sh dev_scripts/run_mlflow_server.sh
# Access: http://127.0.0.0:5000
```

## 2. Pipeline in Airlfow
**To launch Airflow Server:**
1. Download `docker-compose.yaml` file from the official site:
```bash
curl -LfO https://airflow.apache.org/docs/apache-airflow/2.7.3/docker-compose.yaml
```
2. Open the file with any editor, comment out line 53 and remove comment from line 54.
3. Save aurflow_uid to environmental variables:
```bash
echo -e "\nAIRFLOW_UID=$(id -u)" >> .env
```
4. Create container with Airflow and launch it:
```bash
cd airflow_service
docker compose up airflow-init
docker compose down --volumes --remove-orphans
docker compose up --build
# Access: http://127.0.0.0:8080/
```
5. To stop service:
```bash
cd airflow_service
docker compose down
```

**Airflow DAG Description**
|DAG|Description|Schedule|
|:---|:---|:---|
|1_prepare_dataset.py|Contains following steps: extract raw data, transform and load prepared datasets (`items`, `events`) to the database.|Daily, at 22:00|
|2_train_als_model.py|Contains following steps: extract `items` and `events` datasets, train encoders, encode user and item ids, save encoders to S3, train ALS model and save it to S3.|Monthly, on 25th, at 22:15|
|3_prepare_default_recommendations.py|Contains following steps: extract prepared data, create default recommendations, load them to the database.|Daily, at 22:30|
|4_prepare_personalized_recommendations.py|Contains following steps: extract prepared data, create personalized recommendations for all eligible users using ALS model, load them to the database.|Daily, at 22:45|
|5_prepare_i2i_recommendations.py|Contains following steps: extract prepared data, create similar item recommendations for all eligible users using ALS model, load them to the database.|Daily, at 23:00|
|6_prepare_similar_items.py|Contains following steps: extract prepared data, create DataFrame with similar items for all unique items, load them to the database.|Daily, at 23:00|
|7_train_ranking_model_and_rank_recommendations.py|Contains following steps: extract prepared data and trained encoders, create new features, prepare candidates for ranking, train ranking model (CatBoost) and save it to S3, create ranked recommendations and load them to the database.|Monthly, on 25th, at 23:45|

All DAGs have Telegram callback option.

## 3. Fast API service
Launch service without containerization:
```bash
cd fast_api_service
uvicorn recsys_app.recsys_app:app --reload --port 8081 --host 0.0.0.0
```

Launch service in Docker container with monitoring:
```bash
cd fast_api_service
docker compose up --build
```

To test the service:
```bash
cd fast_api_service
python app_test.py
```
Test log will be saved in the `fast_api_service/test_service.log` file.

**Monitoring metrics**
|Metrics|Description|
|:---|:---|
|RAM Usage (MB)|Tracks the memory usage of the container in megabytes. This infrastructure-level metric helps monitor virtual RAM (vRAM) consumption within the container. If memory usage approaches the container's allocated limit, it could result in out-of-memory (OOM) errors, causing the container to stop and potentially leading to service downtime.|
|Request Processing Time|Measures the time taken to process a single HTTP request, recorded in seconds. This metric provides insights into the performance of the service, helping to identify slow endpoints or potential bottlenecks in request handling. 95 percentille is used to track mean request processing time.|
|Requests by Status|Tracks the total number of HTTP requests received by the service, categorized by status codes (e.g., 200, 404, 500). This metric provides visibility into request success rates and error patterns.|
|Number of requests of default recommendations|Counts the total number of requests for default recommendations. This metric helps track usage patterns and demand for the default recommendations.|
|Number of requests of personalized recommendations|Counts the total number of requests for personalized recommendations. This metric highlights the usage and demand for the personalized recommendations.|
|Number of requests of online (i2i) recommendations|Counts the total number of requests for online item-to-item (i2i) recommendations. This metric is useful for understanding the demand for real-time recommendations.|