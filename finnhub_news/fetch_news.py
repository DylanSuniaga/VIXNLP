import requests
import pandas as pd
from datetime import datetime
import time
import os

# API parameters
API_KEY = "cvqnit1r01qp88cmnod0cvqnit1r01qp88cmnodg"
BASE_URL = "https://finnhub.io/api/v1/company-news"
FROM_DATE = "2025-01-15"
TO_DATE = "2025-02-20"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "GS", "BAC", "WFC", "TSLA", "BA", "CAT", "XOM", "CVX"]

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create empty list to store all articles
all_articles = []

# Loop through each ticker
for ticker in TICKERS:
    print(f"Fetching news for {ticker}...")
    
    # Define parameters for API request
    params = {
        "symbol": ticker,
        "from": FROM_DATE,
        "to": TO_DATE,
        "token": API_KEY
    }
    
    try:
        # Send GET request to Finnhub API
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Extract articles from response
        articles = response.json()
        
        # Process each article
        for article in articles:
            # Extract required fields
            processed_article = {
                "date": datetime.fromtimestamp(article["datetime"]).date(),
                "symbol": ticker,
                "headline": article["headline"],
                "summary": article["summary"],
                "source": article["source"],
                "url": article["url"]
            }
            
            # Add to our list
            all_articles.append(processed_article)
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        continue

# Convert list of articles to DataFrame
news_df = pd.DataFrame(all_articles)

# Save DataFrame to CSV
csv_path = os.path.join("data", "finnhub_news.csv")
news_df.to_csv(csv_path, index=False)
print(f"\nData saved to {csv_path}")

# Print the resulting DataFrame
print("\nSample of collected news articles:")
print(news_df.head())

# Display number of articles per ticker
print("\nNumber of articles per ticker:")
ticker_counts = news_df["symbol"].value_counts()
print(ticker_counts)

print(f"\nTotal articles collected: {len(news_df)}") 