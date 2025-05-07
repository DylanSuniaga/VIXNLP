import requests
import csv
import os
import time
from datetime import datetime, timedelta

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv is not installed. Using environment variables directly.")
    print("To install: pip install python-dotenv")

def fetch_news_articles(api_key, topic, limit=10, countries=None, languages=None, date=None, offset=0):
    """
    Fetch news articles from the mediastack API.
    
    Args:
        api_key (str): The API key for accessing mediastack
        topic (str): The topic to search for
        limit (int): Maximum number of articles to return
        countries (str): Optional comma-separated list of country codes
        languages (str): Optional comma-separated list of language codes
        date (str): Optional date in YYYY-MM-DD format or date range
        offset (int): Optional pagination offset
    
    Returns:
        list: List of articles or None on error
    """
    # Base URL for the mediastack API
    base_url = "http://api.mediastack.com/v1/news"
    
    # Set up the request parameters
    params = {
        "access_key": api_key,
        "keywords": topic,
        "limit": limit,
        "offset": offset
    }
    
    # Add date parameter if provided
    if date:
        params["date"] = date
    
    # Add optional filters if provided
    if countries:
        params["countries"] = countries
    
    if languages:
        params["languages"] = languages
    
    try:
        date_info = f" for date {date}" if date else ""
        print(f"Fetching up to {limit} news articles about '{topic}'{date_info}...")
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Check if the API returned an error
            if 'error' in data:
                print(f"API Error: {data['error']['info']}")
                return None
            
            # Extract the articles from the response
            articles = data.get('data', [])
            pagination = data.get('pagination', {})
            total = pagination.get('total', 0)
            
            print(f"Found {len(articles)} articles for topic '{topic}'{date_info}. Total available: {total}")
            
            # Add the topic to each article
            for article in articles:
                article['topic'] = topic
                
            return {
                'articles': articles,
                'total': total,
                'has_more': len(articles) + offset < total
            }
        else:
            print(f"Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def fetch_all_articles_for_date_range(api_key, topic, start_date, end_date, limit=100, countries=None, languages=None, max_articles=1000):
    """
    Fetch all articles for a date range, handling pagination.
    
    Args:
        api_key (str): The API key for accessing mediastack
        topic (str): The topic to search for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        limit (int): Maximum number of articles per request
        countries (str): Optional comma-separated list of country codes
        languages (str): Optional comma-separated list of language codes
        max_articles (int): Maximum total articles to fetch (default: 1000)
    
    Returns:
        list: List of all articles found
    """
    # Format the date parameter
    date_param = f"{start_date},{end_date}"
    
    all_articles = []
    offset = 0
    has_more = True
    
    while has_more and len(all_articles) < max_articles:
        result = fetch_news_articles(
            api_key=api_key,
            topic=topic,
            limit=limit,
            countries=countries,
            languages=languages,
            date=date_param,
            offset=offset
        )
        
        if not result or not result['articles']:
            break
        
        # Only add articles up to the max_articles limit
        remaining = max_articles - len(all_articles)
        all_articles.extend(result['articles'][:remaining])
        
        # Check if we need to paginate
        has_more = result['has_more'] and len(all_articles) < max_articles
        offset += limit
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(1.5)
        
        if len(all_articles) >= max_articles:
            print(f"Reached maximum article limit ({max_articles}) for topic '{topic}'")
    
    return all_articles

def save_to_csv(all_articles, filename):
    """
    Save the articles to a CSV file.
    
    Args:
        all_articles (list): List of article dictionaries
        filename (str): Name of the CSV file to create
    """
    # Define the fields we want to extract from each article
    fields = ['topic', 'title', 'description', 'source', 'published_at', 'url']
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Create a CSV writer
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            
            # Write the header row
            writer.writeheader()
            
            # Write each article as a row
            for article in all_articles:
                # Extract only the fields we want
                row = {field: article.get(field, '') for field in fields}
                writer.writerow(row)
            
        print(f"Data successfully saved to {filename}")
        print(f"Total articles saved: {len(all_articles)}")
    
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

def get_date_ranges(start_date, end_date, interval_days=90):
    """
    Split a date range into smaller chunks to optimize API calls.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        interval_days (int): Maximum number of days per chunk
    
    Returns:
        list: List of tuples containing (start_date, end_date) for each chunk
    """
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_ranges = []
    current_start = start
    
    while current_start < end:
        # Calculate the end of this chunk
        current_end = current_start + timedelta(days=interval_days)
        
        # If we've gone past the overall end date, use that instead
        if current_end > end:
            current_end = end
        
        # Add this chunk to our list
        date_ranges.append((
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        ))
        
        # Move to the next chunk
        current_start = current_end + timedelta(days=1)
    
    return date_ranges

def main():
    # API key for mediastack - retrieve from environment variable for security
    api_key = os.environ.get("MEDIASTACK_API_KEY")
    
    if not api_key:
        print("Error: MEDIASTACK_API_KEY environment variable not set.")
        print("Please set it with: export MEDIASTACK_API_KEY='your_api_key'")
        return
    
    # Date range to fetch news for (MM/DD/YYYY to YYYY-MM-DD)
    start_date = "2025-04-16"
    end_date = "2025-05-06"
    
    # List of topics that typically influence the VIX
    vix_topics = [
        "war",
        "recession", 
        "tax",                      # instead of "taxes"
        "monetary",
        "geopolitical",
        "inflation",
        "trade",                    # instead of "global trade disputes"
        "disaster",                 # instead of "natural disasters"
        "interest rate",
        "federal reserve",
        "tariff",
        "stock market",
        "economy" 
    ]

    
    # Number of articles to fetch per request (API maximum is 100)
    articles_per_request = 100
    
    # Maximum articles to fetch per topic and date range
    max_articles_per_topic = 1000
    
    # Optional filters
    countries = "us"
    languages = "en"
    
    # List to store all articles
    all_articles = []
    
    # Get date ranges for API calls
    date_ranges = get_date_ranges(start_date, end_date, interval_days=90)
    print(f"Splitting date range into {len(date_ranges)} chunks for optimization")
    
    # Iterate over each topic
    for topic in vix_topics:
        topic_articles = []
        
        # Process each date range for this topic
        for range_start, range_end in date_ranges:
            print(f"Processing date range {range_start} to {range_end} for topic '{topic}'")
            
            # Fetch articles for the current topic and date range
            articles = fetch_all_articles_for_date_range(
                api_key=api_key,
                topic=topic,
                start_date=range_start,
                end_date=range_end,
                limit=articles_per_request,
                countries=countries,
                languages=languages,
                max_articles=max_articles_per_topic
            )
            
            # If articles were found, add them to our topic list
            if articles:
                topic_articles.extend(articles)
                print(f"Total articles collected for '{topic}' so far: {len(topic_articles)}")
            else:
                print(f"No articles found for topic '{topic}' in range {range_start} to {range_end} or API request failed.")
            
            # Add a small delay to avoid hitting API rate limits
        
        print(f"Collected {len(topic_articles)} articles for topic '{topic}'")
        all_articles.extend(topic_articles)
    
    # Save all collected articles to a single CSV file
    if all_articles:
        # Define the output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "data"
        output_file = os.path.join(output_dir, f"vix_news_{timestamp}.csv")
        
        # Save the articles to the CSV file
        save_to_csv(all_articles, output_file)
        
        print(f"Data collection complete. Collected {len(all_articles)} articles across {len(vix_topics)} topics.")
    else:
        print("No articles to save.")

if __name__ == "__main__":
    main() 