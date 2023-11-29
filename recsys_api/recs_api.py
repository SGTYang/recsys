from typing import Union
from fastapi import FastAPI

from cassandra_api import Cassandra

app = FastAPI()

TOP_N = 10

@app.get("/recommender/{username}")
async def get_recommends(username: str):
    res = []
    cassandra_obj = Cassandra()
    
    user_rating_hist = cassandra_obj.load_rating(username)

    # Calculate score here
    user_rating_hist[:min(len(user_rating_hist), 5)]
    
    # similarity_test table's scheme : user_id, similarity, target_id
    similarity_query = "SELECT * FROM user_profile_test.similarity_test WHERE user_id=%s"
    similarity_query_futures = [session.execute_async(similarity_query, [target_id]) for target_id in user_rating_hist[:min(len(user_rating_hist), 5)]]

    # wait for them to complete and use the results
    for future in similarity_query_futures:
        rows = future.result()
        for _, _, similar_user_id in rows:
            if len(res) == TOP_N:
                break
            # TODO: Add filter function
            if filter(similar_user_id):
                continue
            res.append(similar_user_id)

    return res

def filter(similar_user_id):
    if similar_user_id in ["History"]:
        return True