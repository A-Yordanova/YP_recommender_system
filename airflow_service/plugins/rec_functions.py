import pandas as pd

def get_available_items(items_df):
    available_items = items_df[items_df["is_available"]==1]["item_id"].tolist()
    return available_items
            
def get_user_history(user_id, events_df):
    interacted_items = events_df[events_df["user_id"]==user_id]["item_id"].unique().tolist()
    return interacted_items

def get_eligible_users(events_df, threshold):
    all_users = events_df["user_id"].unique().tolist()
    eligible_users = []
    for user in all_users:
        interacted_items = get_user_history(user, events_df)
        if len(interacted_items) >= threshold:
            eligible_users.append(user)
    return eligible_users