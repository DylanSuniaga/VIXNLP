# VIXNLP: Volatility Index Prediction with Natural Language Processing

This project explores the relationship between financial news sentiment and market volatility (VIX). It uses natural language processing to extract sentiment from news headlines and descriptions, then applies time series analysis and machine learning techniques to predict volatility regimes and VIX levels.

## Project Overview

The VIXNLP project consists of several integrated components:

1. **News Data Collection**: Methods for gathering financial news from multiple sources
2. **Sentiment Analysis**: NLP models for extracting sentiment from news text
3. **Feature Engineering**: Creating predictive features from news sentiment
4. **Volatility Prediction**: Models for forecasting VIX movements based on news sentiment

## Data Pipeline

```
News Sources → Data Collection → Raw News Data → Sentiment Analysis → Feature Engineering → Volatility Models
```

## Components

### 1. News Collection

The project includes multiple methods to collect financial news:

- **Finnhub API Integration**: Programmatic access to company-specific news
- **Historical News Dataset**: 7-year macro news history dataset for backtesting
- **MediaStack API**: Script for accessing MediaStack's news database

### 2. Sentiment Analysis

We apply NLP techniques to extract sentiment from news:

- **FinBERT Model**: Financial domain-specific BERT model for sentiment scoring
- **Headline vs. Summary Analysis**: Different sentiment extraction for headlines and content
- **Sentiment Calibration**: Adjusting sentiment scores based on financial domain knowledge

### 3. Volatility Prediction Models

The project implements multiple modeling approaches:

#### Markov Regime-Switching Model

This model identifies different market states (normal and panic regimes) and predicts VIX levels:

$$
\text{VIX}_{t+1} = 
\begin{cases}
\alpha_0 + \beta_0^1 X_1 + \beta_0^2 X_2 + \epsilon_t, & \text{if } S_t = 0 \text{ (Normal regime)} \\
\alpha_1 + \beta_1^1 X_1 + \beta_1^2 X_2 + \epsilon_t, & \text{if } S_t = 1 \text{ (Panic regime)}
\end{cases}
$$

Where:
- $X_1$ represents mean sentiment decay (previous day's sentiment)
- $X_2$ represents negative shock intensity
- $\epsilon_t$ is the error term
- $S_t$ is the market regime state

#### Random Forest Classification Model

This model predicts regime transitions with features:

$$P(S_{t+1} = 1) = f(\text{sentiment\_features})$$

Features include:
- Mean sentiment statistics
- Sentiment extremes (shocks)
- Sentiment dominance metrics
- Regime history

## Results

Our analysis has identified:

1. Two distinct market volatility regimes with different baseline VIX levels
2. Strong influence of negative news shocks on regime transitions
3. Predictive power of previous day's sentiment on next-day volatility
4. Classification accuracy of 94% for regime prediction

## Project Structure

- **/data/**: Collected and processed datasets
- **/finnhub_news/**: Finnhub API integration
- **/utils/**: Utility functions and helpers
- **mediastack_news.py**: Script for accessing MediaStack's news API
- **clean_csv.py**: Utility for cleaning and processing CSV data files

## Detailed Documentation

For more detailed information, see the component-specific documentation:

- [Finnhub News Fetcher](finnhub_news/README.md): Details on API integration and sentiment scoring

## Usage Notes

### Data Size Management

Due to GitHub file size limits, large data files are ignored in this repository. The `.gitignore` file excludes:
- All CSV files in the `/data/` directory
- All CSV files in `/finnhub_news/data/`
- Files that start with "clean"
- Specific files like "news.csv" and "vix_news.csv"

When working with this repository, you'll need to run the data collection scripts to generate these files locally.

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

3. Set up API keys:
   - Create a `.env` file in the project root with the following:
   ```
   MEDIASTACK_API_KEY=your_mediastack_api_key
   FINNHUB_API_KEY=your_finnhub_api_key
   ```
   - You'll need to register for free API keys at:
     - [MediaStack](https://mediastack.com/)
     - [Finnhub](https://finnhub.io/)
   
   - To load these environment variables, install python-dotenv:
   ```
   pip install python-dotenv
   ```
   - Then add this to the top of your Python scripts:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```