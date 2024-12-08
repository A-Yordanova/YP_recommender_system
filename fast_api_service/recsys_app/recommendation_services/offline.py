import logging
import pandas as pd
from prometheus_client import Counter
from recsys_app.utility_scripts.database import create_postgresql_connection

logger = logging.getLogger(__name__)
logger.propagate = True

request_default_count = Counter("recommendation_default_requests_total", "Total number of default recommendation requests")
request_personalized_count = Counter("recommendation_personalized_requests_total", "Total number of personalized recommendation requests")

class PreparedRecommendations:
    def __init__(self, db_connection):
        self.recs = {"default": None, "personalized": None}
        #self.stats = {
        #    "request_default_count": 0,
        #    "request_personalized_count": 0
        #}
        self.db_connection = db_connection

    def load(self, type, query: str, **kwargs):
        """
        Loads recommendations from the database.
        """
        logger.info(f"Loading recommendations, type: {type}.")
        try:
            self.recs[type] = pd.read_sql(query, self.db_connection, index_col="index", **kwargs)
            logger.info(f"Recommendations loaded for type {type}: {self.recs[type].head(1)}")
        except Exception as e:
            logger.error(f"Error loading recommendations for type {type}: {e}")

    def get(self, user_id: int, k: int = 10):
        """
        Returns a list of recommendations for a user.
        """
        try:
            recs_offline = self.recs["personalized"].query(f"user_id == {user_id}")
            if recs_offline.empty:
                logger.warning(f"No prepared recommendations found for user {user_id}. Using default.")
                recs_offline = self.recs["default"]
                recs_offline = recs_offline["item_id"].to_list()[:k]
                #self.stats["request_default_count"] += 1
                request_default_count.inc()
            else:
                recs_offline = recs_offline["item_id"].to_list()[:k]
                #self.stats["request_personalized_count"] += 1
                request_personalized_count.inc()
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            recs_offline = []

        return recs_offline

    #def stats(self):
    #    logger.info("Stats for recommendations")
    #    for name, value in self.stats.items():
     #       logger.info(f"{name:<30} {value}")