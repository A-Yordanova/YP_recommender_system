import logging
import pandas as pd
from prometheus_client import Counter
from recsys_app.utility_scripts.database import create_postgresql_connection

logger = logging.getLogger(__name__)
logger.propagate = True

request_i2i_count = Counter("recommendation_i2i_requests_total", "Total number of i2i recommendation requests")

class EventStore:

    def __init__(self, max_events_per_user: int = 3):
        self.events = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id, item_id):
        """
        Save interactions
        """
        user_events = self.events.get(user_id, [])
        self.events[user_id] = [item_id] + user_events[:self.max_events_per_user]

    def get(self, user_id, k):
        """
        Return user history of interactios
        """
        user_events = self.events.get(user_id, [])[:k]

        return user_events

class SimilarItems:
    def __init__(self, db_connection):
        self.similar_items = None
        self.stats = {
            "request_similar_items_count": 0
        }
        self.db_connection = db_connection

    def load(self, query: str, **kwargs):
        """
        Loads recommendations from the database.
        """
        logger.info(f"Loading recommendations, type: online.")
        self.similar_items = pd.read_sql(query, self.db_connection, index_col="index", **kwargs)
        logger.info(f"Recommendations loaded for type online: {self.similar_items.head(1)}")

    def get(self, item_id: int, k: int = 5):
        """
        Returns a list of similar items for a given item_id
        """  
        try:
            recs_online = self.similar_items[self.similar_items["item_id"] == item_id].nlargest(k, "score")
            recs_online = {"similar_item_id": recs_online["similar_item_id"].tolist(), "score": recs_online["score"].tolist()}
            request_i2i_count.inc()
        except KeyError:
            logger.error(f"No similar items found for {item_id}")
            recs_online = {"similar_item_id": [], "score": []}
        return recs_online