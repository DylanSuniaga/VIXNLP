#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import logging
import time
import random
import re
from datetime import datetime

class VIXPoliticsNewsAggregator:
    """Class to aggregate VIX-related political news from multiple sources"""
    
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Common headers for requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Keywords to filter for VIX-related content
        self.vix_keywords = [
            'vix', 'volatility index', 'market volatility', 'fear index', 'fear gauge',
            'cboe', 'chicago board options exchange', 'volatility'
        ]
        
        # Keywords specifically for politics and policy
        self.politics_keywords = [
            'policy', 'government', 'president', 'biden', 'trump', 'congress', 'senate',
            'house', 'election', 'vote', 'regulation', 'tariff', 'tax', 'legislation',
            'geopolitical', 'political', 'administration', 'fiscal', 'budget', 'deficit',
            'treasury secretary', 'federal', 'central bank', 'sec', 'securities', 'democrat',
            'republican', 'policy maker', 'washington', 'white house', 'capitol', 'diplomacy',
            'diplomatic', 'sanctions', 'powell', 'fed chair', 'supreme court', 'federal reserve',
            'campaign', 'midterm', 'presidential', 'governor', 'nominee', 'lawmaker'
        ]
        
        # Terms that indicate the text is likely not a news headline
        self.non_headline_terms = [
            'account', 'settings', 'login', 'sign in', 'sign up', 'register', 'subscribe',
            'password', 'watchlist', 'recently viewed', 'search', 'menu', 'navigation',
            'advertisement', 'sponsored', 'partner', 'upgrade', 'privacy', 'terms of use',
            'cookie', 'contact us', 'about us', 'help', 'support', 'video center'
        ]
    
    def is_political_vix_title(self, title):
        """
        Check if a title is relevant to VIX and politics
        
        Args:
            title (str): The title to check
            
        Returns:
            bool: True if relevant, False otherwise
        """
        title_lower = title.lower()
        
        # Check if title contains any terms indicating it's not a headline
        if any(term in title_lower for term in self.non_headline_terms):
            return False
            
        # Check if title is too short to be a real headline
        if len(title) < 20:
            return False
        
        # Check if title contains any VIX-related keywords
        vix_relevant = any(keyword in title_lower for keyword in self.vix_keywords)
        
        # Check if title contains politics keywords
        politics_relevant = any(keyword in title_lower for keyword in self.politics_keywords)
        
        # Title must be related to both VIX AND politics
        return vix_relevant and politics_relevant
    
    def get_political_headlines(self, url, site_name, selectors=None, additional_filters=None):
        """Generic method to get political VIX headlines from a site
        
        Args:
            url (str): URL to scrape
            site_name (str): Name of the site for logging
            selectors (list): CSS selectors to use for finding headlines
            additional_filters (func): Optional additional filtering function
            
        Returns:
            list: Filtered headlines
        """
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Send request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Default selectors if none provided
            if not selectors:
                selectors = ['h1', 'h2', 'h3', 'h4', '.headline', '[class*="headline"]', '[class*="title"]']
            
            # Get all headlines using selectors
            for selector in selectors:
                for element in soup.select(selector):
                    # Skip navigation elements
                    if element.find_parent('nav') or element.find_parent('header') or element.find_parent('footer'):
                        continue
                        
                    headline_text = element.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # Apply additional filters if provided
            if additional_filters and callable(additional_filters):
                titles = additional_filters(titles)
            
            # Filter for VIX-related political titles
            vix_political_titles = [title for title in titles if self.is_political_vix_title(title)]
            
            logging.info(f"Found {len(vix_political_titles)} VIX-related political titles from {site_name}")
            return vix_political_titles
            
        except Exception as e:
            logging.error(f"Error fetching {site_name} VIX political titles: {e}")
            return []
    
    def get_thehill_titles(self):
        """Scrape VIX-related political titles from The Hill"""
        selectors = ['.story-headline', '.title-headline', '.headline', '.related-content__heading', 'h2.node-title']
        url = "https://thehill.com/business/"
        return self.get_political_headlines(url, "The Hill", selectors)
    
    def get_marketwatch_politics_titles(self):
        """Scrape VIX-related political titles from MarketWatch"""
        selectors = ['h3.article__headline', '.article__headline', '[class*="headline"]', 'h2']
        url = "https://www.marketwatch.com/economy-politics"
        return self.get_political_headlines(url, "MarketWatch Politics", selectors)
    
    def get_foxbusiness_titles(self):
        """Scrape VIX-related political titles from Fox Business"""
        selectors = ['.title', '.headline', '.article-title', '[class*="headline"]', 'h2.title']
        url = "https://www.foxbusiness.com/politics"
        return self.get_political_headlines(url, "Fox Business", selectors)
        
    def get_yahoo_finance_politics_titles(self):
        """Scrape VIX-related political titles from Yahoo Finance Politics"""
        selectors = ['h3', '.Fw\\(b\\)', '.js-content-viewer', '[class*="headline"]']
        url = "https://finance.yahoo.com/topic/politics/"
        return self.get_political_headlines(url, "Yahoo Finance Politics", selectors)
        
    def get_cnbc_politics_titles(self):
        """Scrape VIX-related political titles from CNBC Politics"""
        selectors = ['.Card-title', '.headline', '.headline__text', 'a.Card-title']
        url = "https://www.cnbc.com/politics/"
        return self.get_political_headlines(url, "CNBC Politics", selectors)
    
    def get_all_vix_politics_titles(self):
        """Aggregate VIX-related political titles from all sources"""
        all_titles = {}
        
        # Get titles from each source
        all_titles['The Hill'] = self.get_thehill_titles()
        all_titles['MarketWatch Politics'] = self.get_marketwatch_politics_titles()
        all_titles['Fox Business'] = self.get_foxbusiness_titles()
        all_titles['Yahoo Finance Politics'] = self.get_yahoo_finance_politics_titles()
        all_titles['CNBC Politics'] = self.get_cnbc_politics_titles()
        
        return all_titles

def main():
    """Main function to run the VIX political news aggregator"""
    aggregator = VIXPoliticsNewsAggregator()
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get all VIX-related political titles
    all_titles = aggregator.get_all_vix_politics_titles()
    
    # Print results
    print(f"\nVIX-Related Political News ({timestamp}):")
    print("=" * 80)
    
    total_titles = 0
    
    for source, titles in all_titles.items():
        if titles:
            print(f"\n{source} ({len(titles)} articles):")
            print("-" * 50)
            for i, title in enumerate(titles, 1):
                print(f"{i}. {title}")
            total_titles += len(titles)
        else:
            print(f"\n{source}: No VIX-related political titles found")
    
    print("\n" + "=" * 80)
    print(f"Total VIX-related political articles found: {total_titles}")
    
    if total_titles == 0:
        print("\nNOTE: VIX-related political news can be cyclical. Political impact on markets")
        print("tends to be more widely reported during elections, major policy announcements,")
        print("or periods of significant market volatility caused by political events.")
        print("Try running during periods of political uncertainty or market volatility.")

if __name__ == "__main__":
    main() 