#!/usr/bin/env python3
"""
NOTE: This scraper is no longer used due to Reuters' anti-scraping measures.
The site consistently returns 401 HTTP Forbidden responses to our scraping attempts.
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
    print(f"\nReuters Titles Scraper ({timestamp}):")
    print("-" * 50)
    print("This scraper is no longer active due to Reuters' anti-scraping measures.")
    print("Reuters consistently returns 401 HTTP Forbidden responses to scraping attempts.")
    print("\nAlternatives:")
    print("1. Use a proxy service or commercial scraping solution")
    print("2. Check if Reuters offers an official API")
    print("3. Use other financial news sources that allow scraping")

if __name__ == "__main__":
    main() 