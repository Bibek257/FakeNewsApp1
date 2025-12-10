# src/agent/ai_agent.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import requests
from sentence_transformers import SentenceTransformer, util
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


class AIAgent:
    def __init__(self, api_key: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the AI Agent with a News API key and a Sentence Transformer model.
        """
        self.api_key = api_key
        self.model = SentenceTransformer(model_name)

    def search_news(self, query: str, top_k: int = 5):
        """
        Search for news articles related to the query using NewsAPI.
        Then rank them based on semantic similarity with the query.
        """
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
        response = requests.get(url).json()

        if "articles" not in response:
            return []

        articles = response["articles"]
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        results = []
        for art in articles:
            title = art.get("title", "")
            description = art.get("description", "") or ""
            link = art.get("url", "")
            source = art.get("source", {}).get("name", "Unknown")

            # Combine title + description for better context
            text = f"{title}. {description}"
            art_embedding = self.model.encode(text, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, art_embedding).item()

            results.append({
                "title": title,
                "description": description,
                "url": link,
                "source": source,
                "similarity": round(similarity, 3)
            })

        # Sort by similarity score
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    
    # Example usage
    config = read_yaml("config/urls_config.yaml")
    # getting api key from config file
    
    NEWS_API_KEY = config['ai_agent']['news_api_key'] 
    agent = AIAgent(api_key=NEWS_API_KEY)

    query = "AI in healthcare"
    top_articles = agent.search_news(query, top_k=3)

    for idx, art in enumerate(top_articles, 1):
        print(f"{idx}. {art['title']} ({art['similarity']})")
        print(f"   Source: {art['source']}")
        print(f"   Link: {art['url']}\n")
