#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import logging
import time
import random
from datetime import datetime

def fetch_bbc_titles():
    """
    Scrape article titles from BBC News page.
    
    Returns:
        list: A list of article titles
    """
    url = "https://www.bbc.com/news"
    
    # Use realistic browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    
    try:
        # Add a random delay to simulate human behavior
        time.sleep(random.uniform(1, 3))
        
        # Send request to the target page
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        logging.info(f"Successfully retrieved page with status code: {response.status_code}")
        
        # Parse the HTML content using html5lib parser
        soup = BeautifulSoup(response.text, 'html5lib')
        
        # Find article titles
        headlines = []
        
        # BBC News typically uses various heading elements with specific classes
        # Try various selectors to find headline elements
        
        # Primary approach: Look for headlines in main content
        article_elements = soup.select('.gs-c-promo-heading__title, .gs-c-headline__text, .nw-o-link-split__text')
        for article in article_elements:
            headline_text = article.get_text().strip()
            if headline_text and headline_text not in headlines:
                headlines.append(headline_text)
        
        # Try more generic selectors if needed
        if not headlines:
            # Look for headline elements with specific attributes
            article_elements = soup.select('[data-component="headline"], [data-component="text-block"]')
            for article in article_elements:
                headline_text = article.get_text().strip()
                if headline_text and headline_text not in headlines:
                    headlines.append(headline_text)
        
        # If still no headlines, try a very generic approach
        if not headlines:
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                # Filter to likely article headlines
                if not heading.find_parent('header') and not heading.find_parent('nav'):
                    headline_text = heading.get_text().strip()
                    if headline_text and len(headline_text) > 10 and headline_text not in headlines:
                        headlines.append(headline_text)
        
        # Remove any duplicate headlines and very short ones
        filtered_headlines = []
        for headline in headlines:
            if headline and len(headline) > 10 and headline not in filtered_headlines:
                filtered_headlines.append(headline)
        
        return filtered_headlines
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching BBC titles: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while scraping BBC: {e}")
        return []

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get timestamp for the data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fetch titles
    logging.info("Starting to fetch BBC News titles...")
    titles = fetch_bbc_titles()
    
    # Print results
    print(f"\nBBC News Titles ({timestamp}):")
    print("-" * 50)
    
    if titles:
        for i, title in enumerate(titles, 1):
            print(f"{i}. {title}")
        logging.info(f"Successfully scraped {len(titles)} BBC News titles")
    else:
        print("No titles found or error occurred.")
        logging.warning("No BBC News titles were scraped")
        print("\nNOTE: BBC may be blocking scraping attempts. You might need to:")
        print("1. Use a proxy service")
        print("2. Try a different user agent")
        print("3. Consider if BBC's robots.txt allows scraping")
        print("4. Look into BBC's API if available")

if __name__ == "__main__":
    main() 