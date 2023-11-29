from typing import Union
from fastapi import FastAPI

from cassandra_api import Cassandra

app = FastAPI()

TOP_N = 10

@app.get("/recommender/{username}")
async def get_recommends(username: str):
    res = []
    cassandra_obj = Cassandra()
    
    # load rating user's hist data
    user_rating_hist = cassandra_obj.load_rating(username)

    # load similar users to favored user
    similar_user = cassandra_obj.get_similarity(user_rating_hist.keys())
    
    # calculate score
    res = []
    for fav_user, rating in user_rating_hist.items():
        for target, sims in similar_user[fav_user]:
            res.append((rating * sims, target))
    
    res.sort()

    return res[:TOP_N]

def filter(similar_user_id):
    if similar_user_id in ["History"]:
        return True