"""
X (Twitter) API service for searching tweets.
"""
import requests
from typing import Optional
from config import X_BEARER_TOKEN, X_SEARCH_URL


def search_x_tweets(username: str, query: str = "", max_results: int = 10) -> list:
    """Search for tweets from a specific user on X (Twitter)."""
    if not X_BEARER_TOKEN:
        return []
    
    try:
        # First, get user ID from username
        user_lookup_url = f"https://api.twitter.com/2/users/by/username/{username}"
        user_response = requests.get(
            user_lookup_url,
            headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        )
        
        if user_response.status_code != 200:
            print(f"User lookup failed: {user_response.text}")
            return []
        
        user_id = user_response.json()['data']['id']
        
        # Search for tweets
        search_params = {
            "query": f"from:{username} {query}" if query else f"from:{username}",
            "max_results": max_results,
            "tweet.fields": "created_at,text,author_id"
        }
        
        search_response = requests.get(
            X_SEARCH_URL,
            params=search_params,
            headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        )
        
        if search_response.status_code == 200:
            data = search_response.json()
            return data.get('data', [])
        else:
            print(f"Search failed: {search_response.text}")
            return []
            
    except Exception as e:
        print(f"Error searching X: {e}")
        return []

