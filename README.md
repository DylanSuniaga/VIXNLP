# VIXNLP

## News Title Scrapers

A collection of web scrapers that extract titles from various news websites.

### Setup

1. Create and activate the virtual environment:
   ```
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   .\venv\Scripts\Activate.ps1
   
   # Activate on Mac/Linux
   source venv/bin/activate
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Available Scrapers

- BBC News: Scrapes article titles from BBC News main page
  ```
  python scrapers/titles/bbctitles.py
  ```

- Financial News: Collects general financial, economic, market, and relevant political news from major financial sources (includes headlines that could affect market volatility)
  ```
  python scrapers/titles/financenews.py
  ```

- VIX News: Aggregates business/finance/economics/politics news related to VIX (Volatility Index) from accessible financial sites (MarketWatch, CNBC, Investing.com, Yahoo Finance)
  ```
  python scrapers/titles/vixtitles.py
  ```

- VIX Politics News: Specialized scraper that focuses only on VIX-related political news from accessible sources (The Hill, MarketWatch Politics, Fox Business, Yahoo Finance Politics, CNBC Politics)
  ```
  python scrapers/titles/vixpoliticstitles.py
  ```

### Running All Scrapers

To run all scrapers sequentially:
```
python run_all_scrapers.py
```

This utility script will execute all scrapers in the `scrapers/titles` directory and provide a summary of results.

### Adding New Scrapers

To add a new scraper:
1. Create a new Python file in the `scrapers/titles` directory
2. Name it according to the pattern: `[websitename]titles.py`
3. Follow the existing scraper's pattern for consistency

### Notes

Some websites block scraping attempts. If you encounter 403 Forbidden errors:
- Try different user agents
- Consider using proxy services
- Check the website's robots.txt file
- Look for official APIs if available

We've specifically removed sources that consistently blocked scraping attempts (Bloomberg, Reuters, WSJ, Politico) and focused on accessible sources for reliable results.