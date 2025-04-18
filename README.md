# VIXNLP: Volatility Index Prediction with Natural Language Processing

## 🔍 About This Project

This project demonstrates a VIX and regime forecast pipeline using sentiment signals extracted from real financial news. It includes a complete pipeline from data collection to volatility prediction, with special focus on regime detection and spike prediction.

It includes:
- Daily sentiment ingestion and feature engineering
- Regime-labeled time series construction
- Markov switching models and classifiers
- Spike detection and duration analysis
- Peak level prediction using neural networks
- Half-life analysis for volatility levels
- Final forward regime & volatility forecast

All sensitive keys are excluded. Setup instructions available in `README_SETUP.md`.

## Project Overview

The VIXNLP project consists of several integrated components:

1. **News Data Collection**: Methods for gathering financial news from multiple sources
2. **Sentiment Analysis**: NLP models for extracting sentiment from news text
3. **Feature Engineering**: Creating predictive features from news sentiment
4. **Volatility Prediction**: Models for forecasting VIX movements based on news sentiment
5. **Spike Analysis**: Detection and prediction of volatility spikes and their characteristics

## Data Pipeline

```
News Sources → Data Collection → Raw News Data → Sentiment Analysis → Feature Engineering → Volatility Models → Spike Analysis
```

## Components

### 1. News Collection

The project includes multiple methods to collect financial news:

- **MediaStack API**: Script for accessing MediaStack's news database
- **Historical News Dataset**: 7-year macro news history dataset for backtesting
- **Finnhub API Integration**: Programmatic access to company-specific news

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

#### Spike Analysis

The project includes comprehensive spike analysis:
- Spike detection algorithms
- Duration prediction
- Peak level prediction using neural networks
- Half-life analysis for volatility levels

## Results

Our analysis has identified:

1. Two distinct market volatility regimes with different baseline VIX levels
2. Strong influence of negative news shocks on regime transitions
3. Predictive power of previous day's sentiment on next-day volatility
4. Classification accuracy of 94% for regime prediction
5. Effective spike detection and duration prediction
6. Neural network-based peak level prediction capabilities
7. Half-life analysis for volatility level persistence

## Project Structure

- **/data/**: Collected and processed datasets
- **/finnhub_news/**: Finnhub API integration
- **/utils/**: Utility functions and helpers
- **downloader.ipynb**: Main pipeline notebook containing the complete workflow
- **mediastack_news.py**: Script for accessing MediaStack's news API
- **clean_csv.py**: Utility for cleaning and processing CSV data files

## Usage Notes

### Data Size Management

Due to GitHub file size limits, large data files are ignored in this repository. The `.gitignore` file excludes:
- All CSV files in the `/data/` directory
- All CSV files in `/finnhub_news/data/`
- Files that start with "clean"
- Specific files like "news.csv" and "vix_news.csv"
- All pickle files

When working with this repository, you'll need to run the data collection scripts to generate these files locally.

### Setup

See `README_SETUP.md` for complete setup instructions.

### Running the Pipeline

The complete pipeline is available in `downloader.ipynb`, which includes:
1. Data collection and preprocessing
2. Sentiment analysis
3. Feature engineering
4. Regime detection
5. Spike analysis
6. Prediction models