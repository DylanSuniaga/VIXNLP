# Finnhub News Fetcher

This script fetches company news from the Finnhub API for multiple stock tickers and stores the results in a Pandas DataFrame with sentiment analysis.

## Features

1. **Data Collection**: Fetches financial news articles from Finnhub API for multiple tickers
2. **Sentiment Analysis**: Analyzes the sentiment of headlines and summaries using FinBERT
   - Adds sentiment scores (positive, negative, or neutral) 
   - Scores are adjusted based on sentiment (positive unchanged, negative multiplied by -1, neutral divided by 10)
   - Includes both raw sentiment labels and numeric scores
3. **Data Storage**: Saves processed data to CSV for further analysis

## Requirements

- Python 3.6+
- Required packages listed in `requirements.txt`
  - `requests`: For API calls
  - `pandas`: For data processing
  - `transformers`: For NLP models
  - `torch`: For running the FinBERT model

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:

```bash
python fetch_news.py
```

The script will:
1. Fetch news articles for the specified tickers (AAPL, MSFT, GOOGL, etc.)
2. Extract relevant information (date, headline, summary, source, URL)
3. Create a DataFrame with all collected articles
4. Save the DataFrame to a CSV file in the `data` folder
5. Print a sample of the collected articles and statistics about the number of articles per ticker

## Output

The script saves the collected news articles with sentiment analysis to:
```
data/finnhub_news.csv
```

The CSV includes the following columns:
- `date`: Publication date
- `symbol`: Ticker symbol
- `headline`: Article headline
- `summary`: Article summary
- `source`: News source
- `url`: Link to the article
- `summary_sentiment`: Sentiment score for the summary
- `sentiment_label`: Sentiment label for the summary (positive, negative, neutral)
- `headline_sentiment`: Sentiment score for the headline
- `headline_sentiment_label`: Sentiment label for the headline (positive, negative, neutral)

## Configuration

The following parameters can be modified in the script:
- `API_KEY`: Your Finnhub API key
- `FROM_DATE`: Start date for news articles
- `TO_DATE`: End date for news articles
- `TICKERS`: List of stock tickers to fetch news for 