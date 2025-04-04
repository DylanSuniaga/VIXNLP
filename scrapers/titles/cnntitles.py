#!/usr/bin/env python3
"""
NOTE: This scraper is no longer used due to CNN's anti-scraping measures.
The site periodically blocks scraping attempts with various errors.
"""

import logging
from datetime import datetime

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get timestamp for the data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Print notification
    print(f"\nCNN Titles Scraper ({timestamp}):")
    print("-" * 50)
    print("This scraper is no longer active due to CNN's inconsistent access for scrapers.")
    print("CNN periodically blocks scraping attempts with various errors.")
    print("\nAlternatives:")
    print("1. Use a proxy service or commercial scraping solution")
    print("2. Check if CNN offers an official API")
    print("3. Use other news sources that allow scraping")
    print("4. Note that we still scrape CNBC which covers similar content")

if __name__ == "__main__":
    main() 