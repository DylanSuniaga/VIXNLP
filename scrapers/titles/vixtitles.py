#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import logging
import time
import random
import re
from datetime import datetime

class VIXNewsAggregator:
    """Class to aggregate VIX-related news from multiple sources"""
    
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
        
        # Keywords to filter for VIX and finance/business/economics/politics related content
        self.vix_keywords = [
            'vix', 'volatility index', 'market volatility', 'fear index', 'fear gauge',
            'cboe', 'chicago board options exchange', 'volatility'
        ]
        
        self.finance_keywords = [
            'stock', 'market', 'finance', 'economy', 'economic', 'fed', 'federal reserve',
            'interest rate', 'inflation', 'recession', 'gdp', 'dow', 'nasdaq', 's&p',
            'bond', 'treasury', 'investor', 'investment', 'trading', 'trade', 'hedge',
            'currency', 'forex', 'bull market', 'bear market', 'etf', 'option',
            'derivative', 'portfolio', 'asset', 'equity', 'debt', 'yield', 'earnings'
        ]
        
        self.politics_keywords = [
            'policy', 'government', 'president', 'biden', 'trump', 'congress', 'senate',
            'house', 'election', 'vote', 'regulation', 'tariff', 'tax', 'legislation',
            'geopolitical', 'political', 'administration', 'fiscal', 'budget', 'deficit',
            'treasury secretary', 'federal', 'central bank', 'sec', 'securities', 'democrat',
            'republican', 'policy maker', 'washington'
        ]
        
        # Terms that indicate the text is likely not a news headline
        self.non_headline_terms = [
            'account', 'settings', 'login', 'sign in', 'sign up', 'register', 'subscribe',
            'password', 'watchlist', 'recently viewed', 'search', 'menu', 'navigation',
            'advertisement', 'sponsored', 'partner', 'upgrade', 'privacy', 'terms of use',
            'cookie', 'contact us', 'about us', 'help', 'support', 'video center',
            'trending tickers', 'private companies', 'time to upgrade'
        ]
    
    def is_relevant_title(self, title):
        """
        Check if a title is relevant to VIX and business/finance/politics
        
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
        
        # Check if title contains any finance or politics keywords
        finance_relevant = any(keyword in title_lower for keyword in self.finance_keywords)
        politics_relevant = any(keyword in title_lower for keyword in self.politics_keywords)
        
        # Title must be related to VIX AND (finance OR politics)
        return vix_relevant and (finance_relevant or politics_relevant)
    
    def get_bbc_vix_titles(self):
        """Scrape VIX-related titles from BBC News"""
        url = "https://www.bbc.com/news/business"
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Send request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Get all headlines
            article_elements = soup.select('.gs-c-promo-heading__title, .gs-c-headline__text')
            for article in article_elements:
                headline_text = article.get_text().strip()
                if headline_text and len(headline_text) > 10 and headline_text not in titles:
                    titles.append(headline_text)
            
            # Filter for VIX-related titles
            vix_titles = [title for title in titles if self.is_relevant_title(title)]
            
            logging.info(f"Found {len(vix_titles)} VIX-related titles from BBC")
            return vix_titles
            
        except Exception as e:
            logging.error(f"Error fetching BBC VIX titles: {e}")
            return []
    
    def get_marketwatch_vix_titles(self):
        """Scrape VIX-related titles from MarketWatch"""
        url = "https://www.marketwatch.com/investing/index/vix"
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Send request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Get actual news headlines more precisely
            # Focus on article headlines in the news sections
            for article in soup.select('div.article__content'):
                headline_element = article.select_one('h3.article__headline')
                if headline_element:
                    headline_text = headline_element.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # If the above didn't work, try a more general approach
            if not titles:
                # Look specifically for article headlines, avoiding navigation elements
                for heading in soup.find_all(['h1', 'h2', 'h3']):
                    # Skip elements in typical navigation sections
                    if heading.find_parent('nav') or heading.find_parent('header') or heading.find_parent('footer'):
                        continue
                    
                    # Look for headings in article-like containers
                    if heading.find_parent('article') or 'article' in str(heading.get('class', [])):
                        headline_text = heading.get_text().strip()
                        headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                        if headline_text and len(headline_text) > 15 and headline_text not in titles:
                            titles.append(headline_text)
            
            # Additional filter for MarketWatch to exclude non-headline content
            filtered_titles = []
            for title in titles:
                # Skip nav items, links, etc.
                if not any(term in title.lower() for term in self.non_headline_terms):
                    filtered_titles.append(title)
            
            logging.info(f"Found {len(filtered_titles)} VIX-related titles from MarketWatch")
            return filtered_titles
            
        except Exception as e:
            logging.error(f"Error fetching MarketWatch VIX titles: {e}")
            return []
    
    def get_cnbc_vix_titles(self):
        """Scrape VIX-related titles from CNBC"""
        url = "https://www.cnbc.com/quotes/.VIX"
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Send request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Look specifically for article headlines
            for article in soup.select('div.Card-titleContainer'):
                headline_element = article.select_one('.Card-title')
                if headline_element:
                    headline_text = headline_element.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # If the above didn't find anything, try a more general approach
            if not titles:
                # Look for headings in article-like containers
                for heading in soup.select('a.Card-title, .headline'):
                    headline_text = heading.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # Additional filter for CNBC to exclude non-headline content
            filtered_titles = []
            for title in titles:
                # Skip nav items, links, etc.
                if not any(term in title.lower() for term in self.non_headline_terms):
                    filtered_titles.append(title)
            
            logging.info(f"Found {len(filtered_titles)} VIX-related titles from CNBC")
            return filtered_titles
            
        except Exception as e:
            logging.error(f"Error fetching CNBC VIX titles: {e}")
            return []
    
    def get_investing_com_vix_titles(self):
        """Scrape VIX-related titles from Investing.com"""
        url = "https://www.investing.com/indices/volatility-s-p-500"
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Add additional headers for Investing.com
            headers = self.headers.copy()
            headers["Referer"] = "https://www.investing.com/"
            
            # Send request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Get news headlines
            for article in soup.select('.articleItem'):
                headline_element = article.select_one('.title')
                if headline_element:
                    headline_text = headline_element.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # If no headlines found with the above selector, try a more general approach
            if not titles:
                # Look for article headlines
                for heading in soup.select('.news-link, .news-title, .title a'):
                    headline_text = heading.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # For Investing.com, we're already on the VIX page, so we don't need to filter as strictly
            # Just filter out navigation elements
            filtered_titles = []
            for title in titles:
                # Skip nav items, links, etc.
                if not any(term in title.lower() for term in self.non_headline_terms):
                    filtered_titles.append(title)
            
            logging.info(f"Found {len(filtered_titles)} VIX-related titles from Investing.com")
            return filtered_titles
            
        except Exception as e:
            logging.error(f"Error fetching Investing.com VIX titles: {e}")
            return []
    
    def get_yahoo_finance_vix_titles(self):
        """Scrape VIX-related titles from Yahoo Finance"""
        url = "https://finance.yahoo.com/quote/%5EVIX/"
        titles = []
        
        try:
            # Add a random delay
            time.sleep(random.uniform(1, 2))
            
            # Send request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Get news headlines from the news section
            for article in soup.select('li.js-stream-content'):
                headline_element = article.select_one('h3')
                if headline_element:
                    headline_text = headline_element.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # If no headlines found with the above selector, try a more general approach
            if not titles:
                # Look for article headlines
                for heading in soup.select('h3, .Fw\\(b\\)'):
                    headline_text = heading.get_text().strip()
                    headline_text = re.sub(r'\s+', ' ', headline_text)  # Normalize whitespace
                    if headline_text and len(headline_text) > 15 and headline_text not in titles:
                        titles.append(headline_text)
            
            # Filter out navigation elements
            filtered_titles = []
            for title in titles:
                # Skip nav items, links, etc.
                if not any(term in title.lower() for term in self.non_headline_terms):
                    filtered_titles.append(title)
            
            logging.info(f"Found {len(filtered_titles)} VIX-related titles from Yahoo Finance")
            return filtered_titles
            
        except Exception as e:
            logging.error(f"Error fetching Yahoo Finance VIX titles: {e}")
            return []
    
    def get_all_vix_titles(self):
        """Aggregate VIX-related titles from all sources"""
        all_titles = {}
        
        # Get titles from each source
        all_titles['BBC'] = self.get_bbc_vix_titles()
        all_titles['MarketWatch'] = self.get_marketwatch_vix_titles()
        all_titles['CNBC'] = self.get_cnbc_vix_titles()
        all_titles['Investing.com'] = self.get_investing_com_vix_titles()
        all_titles['Yahoo Finance'] = self.get_yahoo_finance_vix_titles()
        
        return all_titles

def main():
    """Main function to run the VIX news aggregator"""
    aggregator = VIXNewsAggregator()
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get all VIX-related titles
    all_titles = aggregator.get_all_vix_titles()
    
    # Print results
    print(f"\nVIX-Related Business/Finance/Economics/Politics News ({timestamp}):")
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
            print(f"\n{source}: No VIX-related titles found")
    
    print("\n" + "=" * 80)
    print(f"Total VIX-related articles found: {total_titles}")
    
    if total_titles == 0:
        print("\nNOTE: VIX-related news can be cyclical. When market volatility is low,")
        print("there may be fewer articles explicitly mentioning the VIX.")
        print("Try running the scraper during periods of market volatility for more results.")

if __name__ == "__main__":
    main() 