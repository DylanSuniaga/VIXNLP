#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import logging
import time
import random
import re
from datetime import datetime

class FinancialNewsAggregator:
    """Class to aggregate financial, economic, and market news from multiple sources"""
    
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
        
        # Keywords for volatility-related content (including but not limited to VIX)
        self.volatility_keywords = [
            'volatility', 'vix', 'fear index', 'fear gauge', 'market risk',
            'implied volatility', 'options volatility', 'market uncertainty',
            'risk premium', 'risk aversion', 'risk-off', 'risk-on',
            'market swings', 'market gyrations', 'market turbulence',
            'market sentiment', 'investor sentiment', 'market panic',
            'market stress', 'market tension', 'market fear', 'market concern'
        ]
        
        # Keywords for finance, economy, and markets
        self.finance_keywords = [
            'stock', 'market', 'finance', 'economy', 'economic', 'fed', 'federal reserve',
            'interest rate', 'inflation', 'recession', 'gdp', 'dow', 'nasdaq', 's&p',
            'bond', 'treasury', 'investor', 'investment', 'trading', 'trade', 'hedge',
            'currency', 'forex', 'bull market', 'bear market', 'etf', 'option',
            'derivative', 'portfolio', 'asset', 'equity', 'debt', 'yield', 'earnings',
            'monetary policy', 'fiscal policy', 'financial stability', 'market liquidity',
            'credit', 'bank', 'financial institution', 'hedge fund', 'futures', 'commodities',
            'oil price', 'gold price', 'price action', 'technical analysis', 'chart pattern',
            'support level', 'resistance level', 'breakout', 'breakdown', 'rally', 'sell-off',
            'correction', 'crash', 'bubble', 'valuation', 'overvalued', 'undervalued',
            'stimulus', 'bailout', 'quantitative easing', 'qe', 'rate hike', 'rate cut'
        ]
        
        # Keywords for politics with financial implications
        self.politics_keywords = [
            'policy', 'government', 'president', 'biden', 'trump', 'congress', 'senate',
            'house', 'election', 'vote', 'regulation', 'tariff', 'tax', 'legislation',
            'geopolitical', 'political', 'administration', 'fiscal', 'budget', 'deficit',
            'treasury secretary', 'fed chair', 'central bank', 'sec', 'securities',
            'trade war', 'sanctions', 'diplomatic tension', 'geopolitical risk',
            'policy uncertainty', 'regulatory change', 'government shutdown'
        ]
        
        # Terms that indicate the text is likely not a news headline
        self.non_headline_terms = [
            'account', 'settings', 'login', 'sign in', 'sign up', 'register', 'subscribe',
            'password', 'watchlist', 'recently viewed', 'search', 'menu', 'navigation',
            'advertisement', 'sponsored', 'partner', 'upgrade', 'privacy', 'terms of use',
            'cookie', 'contact us', 'about us', 'help', 'support', 'video center'
        ]
    
    def is_relevant_title(self, title):
        """
        Check if a title is relevant to finance, economics, markets, or politics with finance implications
        
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
        if len(title) < 15:
            return False
        
        # Check if title has volatility-related content
        volatility_relevant = any(keyword in title_lower for keyword in self.volatility_keywords)
        
        # Check if title has finance/economics/markets content
        finance_relevant = any(keyword in title_lower for keyword in self.finance_keywords)
        
        # Check if title has politics with financial implications
        politics_relevant = any(keyword in title_lower for keyword in self.politics_keywords)
        
        # Title is relevant if it relates to volatility OR finance OR politics with financial implications
        return volatility_relevant or finance_relevant or politics_relevant
    
    def get_headlines(self, url, site_name, selectors=None, additional_filters=None):
        """Generic method to get finance headlines from a site
        
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
            
            # Filter for relevant titles
            relevant_titles = [title for title in titles if self.is_relevant_title(title)]
            
            logging.info(f"Found {len(relevant_titles)} relevant financial titles from {site_name}")
            return relevant_titles
            
        except Exception as e:
            logging.error(f"Error fetching {site_name} titles: {e}")
            return []
    
    def get_ft_titles(self):
        """Scrape financial titles from Financial Times"""
        selectors = ['.headline', '.js-teaser-heading-link', '.o-teaser__heading', 'h3.o-teaser__heading']
        url = "https://www.ft.com/markets"
        return self.get_headlines(url, "Financial Times", selectors)
    
    def get_marketwatch_titles(self):
        """Scrape financial titles from MarketWatch"""
        selectors = ['h3.article__headline', '.article__headline', 'h3.title', '.story__headline']
        url = "https://www.marketwatch.com/markets"
        return self.get_headlines(url, "MarketWatch", selectors)
    
    def get_yahoo_finance_titles(self):
        """Scrape financial titles from Yahoo Finance"""
        selectors = ['.js-content-viewer', 'h3', '.Mb\\(5px\\)', '[class*="headline"]', '.StretchedBox']
        url = "https://finance.yahoo.com/news/"
        return self.get_headlines(url, "Yahoo Finance", selectors)
    
    def get_cnbc_titles(self):
        """Scrape financial titles from CNBC"""
        selectors = ['.Card-title', '.Card-headline', '.headline', '.headline__text', 'a.Card-title']
        url = "https://www.cnbc.com/finance/"
        return self.get_headlines(url, "CNBC", selectors)
    
    def get_bbc_business_titles(self):
        """Scrape financial titles from BBC Business"""
        selectors = ['.gs-c-promo-heading__title', '.gs-c-headline__text', '.nw-o-link-split__text']
        url = "https://www.bbc.com/news/business"
        return self.get_headlines(url, "BBC Business", selectors)
    
    def get_all_financial_titles(self):
        """Aggregate financial titles from all sources"""
        all_titles = {}
        
        # Get titles from each source
        all_titles['Financial Times'] = self.get_ft_titles()
        all_titles['MarketWatch'] = self.get_marketwatch_titles()
        all_titles['Yahoo Finance'] = self.get_yahoo_finance_titles()
        all_titles['CNBC'] = self.get_cnbc_titles()
        all_titles['BBC Business'] = self.get_bbc_business_titles()
        
        return all_titles

def main():
    """Main function to run the financial news aggregator"""
    aggregator = FinancialNewsAggregator()
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get all financial titles
    all_titles = aggregator.get_all_financial_titles()
    
    # Print results
    print(f"\nFinancial, Economic, and Market News ({timestamp}):")
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
            print(f"\n{source}: No relevant titles found")
    
    print("\n" + "=" * 80)
    print(f"Total financial news articles found: {total_titles}")

if __name__ == "__main__":
    main() 