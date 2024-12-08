import logging

from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from typing import List

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

from recsys_app.recommendation_services.offline import PreparedRecommendations    # Get prepared recommendations from DB (personalized or default)
from recsys_app.recommendation_services.online import SimilarItems, EventStore    # Get user online history and get similar items based on the history

from recsys_app.utility_scripts.database import create_postgresql_connection

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Main app started.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_connection = create_postgresql_connection()
    logger.info("PostgreSQL connection established.")

    global recs_offline_store, recs_online_store, event_store
    recs_offline_store = PreparedRecommendations(db_connection)
    recs_online_store = SimilarItems(db_connection)
    event_store = EventStore(max_events_per_user=3)

    default_recommendations_query = "SELECT index, item_id FROM recsys_default_recommendations"
    personalized_recommendations_query = "SELECT index, user_id, item_id, rank FROM recsys_ranked_recommendations"
    i2i_recommendations_query = "SELECT * FROM recsys_similar_items"
    
    recs_offline_store.load("default", default_recommendations_query)
    recs_offline_store.load("personalized", personalized_recommendations_query)
    recs_online_store.load(i2i_recommendations_query)

    logger.info("Recommendation service started.")
    yield
    logger.info("Recommendation service stopped.")

app = FastAPI(title="Recommendations Service", lifespan=lifespan)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/save_event")
async def save_event(user_id: int, item_id: int):
    """
    Adds user history.
    """
    event_store.put(user_id, item_id)
    return {"success"}

@app.post("/get_user_history")
async def get_user_history(user_id: int, k: int = 10):
    """
    Returns k last interacted items for user_id.
    """
    events = event_store.get(user_id, k)

    return {"events": events}

@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 5):
    """
    Returns list of k offline recommendations for user_id.
    """
    recs = recs_offline_store.get(user_id, k)
    return {"recs": recs}

@app.post("/recommendations_online")
async def recommendations_online(item_ids: List[int] = Query(...), k: int = Query(5)):
    """
    Returns a list of k similar items with highest score.
    """
    if isinstance(item_ids, int):
        item_ids = [item_ids]
        
    def get_similar_items(item_id, k: int = 5):
        try:
            similar_items = recs_online_store.get(item_id, k)
            return similar_items
        except Exception as e:
            logger.error(f"Error fetching similar items for item {item_id}: {e}")
            return {"similar_item_id": [], "score": []}
    
    try:
        all_similar_items = []
        for item_id in item_ids:
            similar_items_for_item_id = get_similar_items(item_id, k)
            for similar_item_id, score in zip(similar_items_for_item_id["similar_item_id"], similar_items_for_item_id["score"]):
                all_similar_items.append({"item_id": similar_item_id, "score": score})

        all_similar_items = sorted(all_similar_items, key=lambda x: x["score"], reverse=True)

        unique_items = {}
        for item in all_similar_items:
            if item["item_id"] not in unique_items:
                unique_items[item["item_id"]] = item["score"]
        
        top_k_items = sorted(unique_items.items(), key=lambda x: x[1], reverse=True)[:k]

        recs = [{"item_id": item_id, "score": score} for item_id, score in top_k_items]
        
    except Exception as e:
        recs = []
        logger.error(f"Error in recommendations_online: {e}")
    
    return {"recs": recs}

def dedup_ids(ids):
    """
    Remove duplicates from recommendations.
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 10):
    """
    Returns blended list of k recommendations for user_id.
    """
    user_history_response = await get_user_history(user_id, k)
    online_history = user_history_response.get("events", [])

    recs_offline = await recommendations_offline(user_id, k)
    recs_offline = list(recs_offline["recs"])

    recs_online_response = await recommendations_online(item_ids=online_history, k=k)
    recs_online = [rec["item_id"] for rec in recs_online_response["recs"]]

    recs_blended = []

    min_length = min(len(recs_offline), len(recs_online))

    for i in range(min_length):
        recs_blended.append(recs_offline[i])
        recs_blended.append(recs_online[i])

    if len(recs_online) > min_length:
        recs_blended.extend(recs_online[min_length:])
    elif len(recs_offline) > min_length:
        recs_blended.extend(recs_offline[min_length:])

    recs_blended = dedup_ids(recs_blended)
    recs_blended = recs_blended[:k]

    return {"recs": recs_blended}