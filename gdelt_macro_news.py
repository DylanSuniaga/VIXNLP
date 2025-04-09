import os
import pandas as pd
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters

def fetch_macro_news(start_date, end_date, keywords):
    """
    Fetches macroeconomic news data from GDELT.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        keywords (list): List of keywords to filter articles
    
    Returns:
        pandas.DataFrame: DataFrame with processed news data
    """
    # Initialize GDELT client
    gdelt = GdeltDoc()
    
    # Convert date strings to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Store all articles
    all_articles = []
    
    # Query GDELT in chunks of 1 day to avoid timeout/size limits
    current_date = start
    while current_date <= end:
        next_date = current_date + timedelta(days=1)
        
        # Format dates for GDELT query
        timespan_start = current_date.strftime('%Y-%m-%d')
        timespan_end = next_date.strftime('%Y-%m-%d')
        
        print(f"Fetching articles from {timespan_start} to {timespan_end}...")
        
        try:
            # Create filters object
            filters = Filters()
            
            # Set the time range
            filters.time_span = timespan_start + " " + timespan_end
            
            # Join keywords with OR for the query
            keyword_query = ' OR '.join(keywords)
            
            # Query GDELT API
            articles = gdelt.article_search(
                keyword_query,
                filters=filters,
                max_articles=250,
                coverage=True,
                tone=True
            )
            
            # Add articles to our collection if results exist
            if articles and 'articles' in articles and articles['articles'] is not None:
                all_articles.extend(articles['articles'])
                print(f"Found {len(articles['articles'])} articles")
            else:
                print("No articles found for this timespan")
                
        except Exception as e:
            print(f"Error fetching data for {timespan_start}: {str(e)}")
        
        # Move to next day
        current_date = next_date
    
    # Convert to DataFrame
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # Select and rename relevant columns
        columns_to_keep = ['date', 'url', 'themes', 'locations', 'tone']
        
        # Check if all columns exist
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        
        if existing_columns:
            df = df[existing_columns]
            
            # Extract specific data from nested structures if available
            if 'locations' in df.columns:
                # Extract country names from locations
                df['countries'] = df['locations'].apply(
                    lambda locs: [loc.get('name', '') for loc in locs] if isinstance(locs, list) else []
                )
            
            if 'tone' in df.columns:
                # Extract tone values
                df['tone_avg'] = df['tone'].apply(
                    lambda t: t.get('tone', 0) if isinstance(t, dict) else 0
                )
                df['tone_positive'] = df['tone'].apply(
                    lambda t: t.get('positive', 0) if isinstance(t, dict) else 0
                )
                df['tone_negative'] = df['tone'].apply(
                    lambda t: t.get('negative', 0) if isinstance(t, dict) else 0
                )
            
            # Format date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        else:
            print("Required columns not found in the API response")
            return pd.DataFrame()
    else:
        print("No articles found")
        return pd.DataFrame()

def main():
    # Define parameters
    start_date = "2025-01-15"
    end_date = "2025-02-20"
    
    # Keywords for macroeconomic relevance
    keywords = [
        "federal reserve", "fed", "inflation", "interest rates", 
        "cpi", "geopolitical", "recession", "conflict", 
        "oil", "macro", "central bank", "hike", "cut"
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Fetch and process news data
    print(f"Fetching macroeconomic news data from {start_date} to {end_date}...")
    news_df = fetch_macro_news(start_date, end_date, keywords)
    
    if not news_df.empty:
        # Save to CSV
        output_path = os.path.join("data", "gdelt_macro_news.csv")
        news_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Total articles collected: {len(news_df)}")
        
        # Display sample
        print("\nSample of collected news articles:")
        print(news_df.head())
    else:
        print("No data to save")

if __name__ == "__main__":
    main() 