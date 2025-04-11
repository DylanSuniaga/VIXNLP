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

# Macro Analysis of Market Volatility and News Sentiment

## Overview
This project analyzes the relationship between financial news sentiment and market volatility (VIX). It uses natural language processing (NLP) to extract sentiment from news headlines and descriptions, then applies time series and machine learning techniques to predict volatility regimes and VIX levels.

## Data Sources
- Financial news data from a CSV file (`vix_news.csv`) containing news headlines, descriptions, and publication dates
- VIX data downloaded automatically within the code

## Methodology

### Sentiment Analysis
The code processes news data in several stages:
1. **Data Loading**: Loads news data with topics, titles, descriptions, sources, and publication dates
2. **Sentiment Extraction**: 
   - Calculates sentiment scores for news descriptions using a summary-based model
   - Calculates sentiment scores for news headlines using a headline-specific model
   - Both models likely use transformer-based NLP techniques

### Feature Engineering
The code creates several sentiment-based features:
- Summary sentiment statistics (mean, min, max, positive/negative counts)
- Headline sentiment statistics (mean, min, max, positive/negative counts)
- Overall sentiment metrics combining both summary and headline sentiment
- Additional engineered features:
  - `mean_sentiment_decay`: Previous day's overall sentiment
  - `shock_pos`: Best (most positive) news of the day
  - `shock_neg`: Worst (most negative) news of the day
  - `sentiment_dominance`: Difference between positive and negative news counts
  - `dominance_trigger`: Binary flag when negative news dominates significantly

### Time Series Analysis
The notebook uses a Markov Regime-Switching Regression model to:
1. Identify different volatility regimes (normal and panic states)
2. Predict VIX levels for the next day based on sentiment features
3. Estimate transition probabilities between regimes

### Classification Model
Additionally, a Random Forest classifier is trained to predict the next day's regime (panic or normal) based on sentiment features.

## Mathematical Models

### Markov Regime-Switching Model
The model assumes that the VIX follows different dynamics in different regimes:

$VIX_{t+1} = \alpha^{S_t} + \beta^{S_t}_1X_1 + \beta^{S_t}_2X_2 + \epsilon_t$

Where:
- $S_t$ is the regime state (0 for normal, 1 for panic)
- $\alpha^{S_t}$ is the regime-dependent intercept
- $\beta^{S_t}_i$ are regime-dependent coefficients
- $X_i$ are sentiment variables (mean_sentiment_decay, shock_neg)
- $\epsilon_t$ is a random error term

The model also estimates transition probabilities between regimes.

## Results and Interpretation

### Identified Regimes
The model identifies two distinct regimes:
- **Regime 0 (Normal)**: Lower baseline VIX level (intercept ~20.1)
- **Regime 1 (Panic)**: Higher baseline VIX level (intercept ~48.2)

### Transition Probabilities
- The probability of staying in Regime 0 (normal) is very high (~97.4%)
- The probability of transitioning from Regime 1 to Regime 0 is low (~5.3%)

### Sentiment Effects
- Negative sentiment shocks appear to have a stronger impact on VIX in the panic regime
- The dominance of negative news can trigger regime shifts
- Mean sentiment decay (previous day's sentiment) influences the next day's VIX level

### Classification Performance
The Random Forest classifier shows:
- High precision and recall for predicting normal regime (0.93 precision, 1.00 recall)
- Good precision but moderate recall for predicting panic regime (1.00 precision, 0.71 recall)
- Overall accuracy of 94%

## Potential Issues and Limitations

1. **Data Timing Issues**: 
   - There appears to be unusual date ranges (2025) in the data, which suggests potential data entry errors or timezone issues

2. **Model Specification**:
   - The Markov Switching model makes strong assumptions about state transitions
   - Limited testing of alternative model specifications
   - Small number of features used in the final model

3. **Overfitting Concerns**:
   - No explicit train/test split shown for the classification model
   - Limited cross-validation to ensure generalizability

4. **Endogeneity Issues**:
   - News sentiment and VIX may have bi-directional causality (high VIX may cause negative news)
   - Potential omitted variable bias (other market factors not included)

5. **Outlier Handling**:
   - No explicit treatment of outliers which could significantly impact regime identification

6. **Regime Stability**:
   - The model assumes only two regimes, which may oversimplify market dynamics
   - Regime transitions might require more complex modeling

## Recommendations for Improvement

1. **Data Quality**:
   - Verify date formats and ranges
   - Consider longer historical periods to capture more regime transitions

2. **Model Validation**:
   - Implement proper train/test splits
   - Use cross-validation to test model stability
   - Compare with alternative models (GARCH, etc.)

3. **Feature Engineering**:
   - Consider including macroeconomic variables
   - Test for non-linear relationships between sentiment and VIX
   - Include market trading volume or other technical indicators

4. **Regime Analysis**:
   - Test models with more than two regimes
   - Consider smooth transition models as alternatives

5. **Causal Analysis**:
   - Implement techniques to address potential endogeneity (instrumental variables)
   - Use Granger causality tests to establish direction of influence

## Usage
To reproduce this analysis:
1. Ensure you have the required packages installed (pandas, numpy, statsmodels, sklearn, matplotlib)
2. Set up the appropriate data file structure with your news data
3. Run the notebook cells in sequence

## Mathematical Representation of Regimes
$$
\text{VIX}_{t+1} = 
\begin{cases}
20.10 - 1.63X_1 - 1.55X_2 + \epsilon_t, & \text{if } S_t = 0 \text{ (Normal regime)} \\
48.19 + 3.33X_1 - 4.23X_2 + \epsilon_t, & \text{if } S_t = 1 \text{ (Panic regime)}
\end{cases}
$$

Where:
- $X_1$ represents mean_sentiment_decay
- $X_2$ represents shock_neg
- $\epsilon_t \sim N(0, 14.12)$
- Transition probabilities: $P(S_t=0|S_{t-1}=0) = 0.974$, $P(S_t=0|S_{t-1}=1) = 0.053$ 