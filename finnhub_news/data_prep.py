import pandas as pd
import numpy as np
from transformers import pipeline
import yfinance as yf
import pandas as pd
from datetime import datetime

pipe = pipeline("text-classification", model="ProsusAI/finbert")

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_macro_df(path):
    macro_df = load_data(path)
    macro_df['date'] = pd.to_datetime(macro_df['published_at'])
    macro_df['date'] = macro_df['date'].dt.date

def sort_by_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date')
    df_sorted = df_sorted.reset_index(drop=True)
    min_date = df['date'].min()
    max_date = df['date'].max()
    return df_sorted, min_date, max_date

def calculate_sentiment(df, col_name, pipe, suffix=''):
    """
    Analyze sentiment of text in a DataFrame column using FinBERT.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the text column to analyze
    col_name : str
        Name of the column containing text to analyze
    pipe : pipeline
        The FinBERT sentiment analysis pipeline
    suffix : str, optional
        Suffix to add to the new column names (default: '')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added sentiment score and label columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Convert column to list first, then praocess each with FinBERT
    texts = result_df[col_name].fillna('').astype(str).tolist()
    
    # Create lists to store sentiment scores and labels
    sentiment_scores = []
    sentiment_labels = []
    
    # Process each text individually
    for text in texts:
        prediction = pipe.predict(text)[0]
        label = prediction['label']
        score = prediction['score']
        
        # Store the label
        sentiment_labels.append(label)
        
        # Calculate the modified score based on label
        if label == 'positive':
            sentiment_scores.append(score)
        elif label == 'negative':
            sentiment_scores.append(score * -1)
        else:  # neutral
            sentiment_scores.append(score / 10)
    
    # Add both the scores and labels to the DataFrame
    result_df[f'{col_name}_sentiment{suffix}'] = sentiment_scores
    result_df[f'{col_name}_sentiment_label{suffix}'] = sentiment_labels
    
    return result_df

def calculate_mean_sentiment(df, start_date, end_date, summary_col, headline_col):
    df['date'] = pd.to_datetime(df['date'])
    filtered_news_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date)
    ]
    daily_sentiment_df = filtered_news_df.groupby(filtered_news_df['date'].dt.date).agg({
        f'{summary_col}': 'mean',
        f'{headline_col}': 'mean'
    }).reset_index()
    daily_sentiment_df = daily_sentiment_df.rename(columns={
        f'{summary_col}': 'avg_summary_sentiment',
        f'{headline_col}': 'avg_headline_sentiment'
    })

    daily_sentiment_df['avg_overall_sentiment'] = (
        daily_sentiment_df['avg_summary_sentiment'] + 
        daily_sentiment_df['avg_headline_sentiment']
    ) / 2

    daily_sentiment_df['date'] = pd.to_datetime(daily_sentiment_df['date'])

    return daily_sentiment_df

def download_vix_data(start_date, end_date):
    vix = yf.Ticker("^VIX")
    vix_df = vix[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Convert index to a date column and reset index
    vix_df["date"] = vix_df.index.date
    vix_df = vix_df.reset_index(drop=True)

    # Reorder columns to have date first
    vix_df = vix_df[["date", "Open", "High", "Low", "Close", "Volume"]]

    vix_df = vix_df[['date', 'Close']]
    vix_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in vix_df.columns]
    vix_df = vix_df[[ 'date_', 'Close_^VIX']].copy()
    vix_df.columns = [ 'date', 'vix_close']

    vix_df['date'] = pd.to_datetime(vix_df['date'])

    return vix_df

def merge_dataframes(df1, df2):
    df1 = df1.sort_values('date')
    df2 = df2.sort_values('date')

    merged_df = pd.merge_asof(
        df1,
        df2,
        on='date',
        direction='forward'
    )
    merged_df = merged_df.rename(columns={'vix_close': 'vix_target'})

    merged_df = merged_df.dropna()
    return merged_df

def load_macro_df(path):
    macro_df = load_data(path)
    macro_df['date'] = pd.to_datetime(macro_df['published_at'])
    macro_df['date'] = macro_df['date'].dt.date
    

def micro_analysis(path, summary_col, headline_col):
    df = load_data(path)
    df, min_date, max_date = sort_by_date(df) #implies dataset is clean, no random dates!!
    df = calculate_sentiment(df, summary_col, pipe, suffix='_summary')
    df = calculate_sentiment(df, headline_col, pipe, suffix='_headline')
    df = calculate_mean_sentiment(df, min_date, max_date, summary_col, headline_col)
    vix_df = download_vix_data(min_date, max_date)
    df = merge_dataframes(df, vix_df)
    return df

def macro_analysis(path, summary_col, headline_col):
    df = load_macro_df(path)
    df, min_date, max_date = sort_by_date(df)
    df = calculate_sentiment(df, summary_col, pipe, suffix='_summary')
    df = calculate_sentiment(df, headline_col, pipe, suffix='_headline')
    df = calculate_mean_sentiment(df, min_date, max_date, summary_col, headline_col)
    vix_df = download_vix_data(min_date, max_date)
    df = merge_dataframes(df, vix_df)
    return df
    