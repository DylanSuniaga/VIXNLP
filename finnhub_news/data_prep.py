import pandas as pd
import numpy as np
from transformers import pipeline
import yfinance as yf
import re
from datetime import datetime
from tqdm import tqdm
import time  # Add import for time module
from dotenv import load_dotenv

load_dotenv()
pipe = pipeline("text-classification", model="ProsusAI/finbert")

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_macro_df(path):
    macro_df = load_data(path)
    
    # Print columns for debugging
    print(f"Available columns in CSV: {macro_df.columns.tolist()}")
    
    # Check if 'published_at' exists, otherwise look for alternatives
    if 'published_at' in macro_df.columns:
        macro_df['date'] = pd.to_datetime(macro_df['published_at'])
    elif 'date' in macro_df.columns:
        macro_df['date'] = pd.to_datetime(macro_df['date'])
    elif 'datetime' in macro_df.columns:
        macro_df['date'] = pd.to_datetime(macro_df['datetime'])
    elif 'time' in macro_df.columns:
        macro_df['date'] = pd.to_datetime(macro_df['time'])
    else:
        # If no date column found, show available columns and raise error
        raise ValueError(f"No date column found in CSV. Available columns: {macro_df.columns.tolist()}")
    
    # Convert to just the date part
    macro_df['date'] = macro_df['date'].dt.date
    return macro_df

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
    
    # Convert column to list first, then process each with FinBERT
    texts = result_df[col_name].fillna('').astype(str).tolist()
    
    # Create lists to store sentiment scores and labels
    sentiment_scores = []
    sentiment_labels = []
    
    # Add progress bar
    print(f"Analyzing sentiment for {len(texts)} entries in column '{col_name}'...")
    
    # Process each text individually with progress bar
    for text in tqdm(texts, desc=f"Sentiment analysis for '{col_name}'"):
        # Clean the text first - remove excessive newlines and whitespace
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate text to avoid exceeding model's token limit (512 tokens max)
        # A simple approximation: truncate to 400 words which should be well below the limit
        truncated_text = ' '.join(clean_text.split()[:400])
        
        try:
            prediction = pipe.predict(truncated_text)[0]
            label = prediction['label']
            score = prediction['score']
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            print(f"Original length: {len(text)}, Clean length: {len(clean_text)}, Truncated length: {len(truncated_text)}")
            # Fallback to neutral sentiment if error occurs
            label = 'neutral'
            score = 0.5
        
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

def calculate_sentiment_stats(df, start_date, end_date, summary_col, headline_col):
    print("Inside calculate_sentiment_stats")
    print(f"Input df shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")
    print(f"Date min: {start_date}, max: {end_date}")
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date
    filtered_news_df = df[
        (df['date'] >= pd.to_datetime(start_date)) & 
        (df['date'] <= pd.to_datetime(end_date))
    ]
    print(f"Filtered df shape: {filtered_news_df.shape}")
    
    # Get the sentiment column names
    summary_sentiment_col = f'{summary_col}_sentiment_summary'
    headline_sentiment_col = f'{headline_col}_sentiment_headline'
    
    print(f"Looking for columns: {summary_sentiment_col} and {headline_sentiment_col}")
    if summary_sentiment_col not in filtered_news_df.columns:
        print(f"WARNING: {summary_sentiment_col} not found in columns!")
    if headline_sentiment_col not in filtered_news_df.columns:
        print(f"WARNING: {headline_sentiment_col} not found in columns!")
    
    # Handle each aggregation separately to avoid MultiIndex complications
    # First, group by date
    grouped = filtered_news_df.groupby(filtered_news_df['date'].dt.date)
    
    # Create a new DataFrame with just the date column first
    result_dates = grouped.size().reset_index()
    result_dates.columns = ['date', 'count']
    
    # Calculate statistics for summary sentiment
    if summary_sentiment_col in filtered_news_df.columns:
        summary_stats = grouped[summary_sentiment_col].agg(['mean', 'min', 'max']).reset_index()
        summary_stats.columns = ['date', 'summary_mean', 'summary_min', 'summary_max']
        result_dates = pd.merge(result_dates, summary_stats, on='date')
        
        # Count positive and negative values
        summary_pos = grouped[summary_sentiment_col].apply(lambda x: (x > 0).sum()).reset_index()
        summary_neg = grouped[summary_sentiment_col].apply(lambda x: (x < 0).sum()).reset_index()
        summary_pos.columns = ['date', 'summary_pos_count']
        summary_neg.columns = ['date', 'summary_neg_count']
        
        result_dates = pd.merge(result_dates, summary_pos, on='date')
        result_dates = pd.merge(result_dates, summary_neg, on='date')
    
    # Calculate statistics for headline sentiment
    if headline_sentiment_col in filtered_news_df.columns:
        headline_stats = grouped[headline_sentiment_col].agg(['mean', 'min', 'max']).reset_index()
        headline_stats.columns = ['date', 'headline_mean', 'headline_min', 'headline_max']
        result_dates = pd.merge(result_dates, headline_stats, on='date')
        
        # Count positive and negative values
        headline_pos = grouped[headline_sentiment_col].apply(lambda x: (x > 0).sum()).reset_index()
        headline_neg = grouped[headline_sentiment_col].apply(lambda x: (x < 0).sum()).reset_index()
        headline_pos.columns = ['date', 'headline_pos_count']
        headline_neg.columns = ['date', 'headline_neg_count']
        
        result_dates = pd.merge(result_dates, headline_pos, on='date')
        result_dates = pd.merge(result_dates, headline_neg, on='date')
    
    # Calculate overall statistics if both columns exist
    if (summary_sentiment_col in filtered_news_df.columns and 
        headline_sentiment_col in filtered_news_df.columns):
        
        for stat in ['mean', 'min', 'max']:
            if f'summary_{stat}' in result_dates.columns and f'headline_{stat}' in result_dates.columns:
                result_dates[f'overall_{stat}'] = (result_dates[f'summary_{stat}'] + result_dates[f'headline_{stat}']) / 2
        
        if 'summary_pos_count' in result_dates.columns and 'headline_pos_count' in result_dates.columns:
            result_dates['overall_pos_count'] = result_dates['summary_pos_count'] + result_dates['headline_pos_count']
            
        if 'summary_neg_count' in result_dates.columns and 'headline_neg_count' in result_dates.columns:
            result_dates['overall_neg_count'] = result_dates['summary_neg_count'] + result_dates['headline_neg_count']
    
    # Convert date to datetime for consistency
    result_dates['date'] = pd.to_datetime(result_dates['date'])
    
    # Drop the count column if not needed
    if 'count' in result_dates.columns:
        result_dates = result_dates.drop(columns=['count'])
    
    print(f"Output columns: {result_dates.columns.tolist()}")
    return result_dates

import time
import pandas as pd
from polygon import RESTClient
import os

def download_vix_data(start_date, end_date, vix_api_key):
    time.sleep(0.5)  # Respectful rate limiting
    
    client = RESTClient(vix_api_key)  # Replace with your actual API key

    aggs = []
    for a in client.list_aggs(
        "I:VIX",
        1,
        "day",
        start_date,
        pd.Timestamp.today().date(),
        sort="asc",
        limit=500
    ):
        aggs.append(a)

    # Convert to DataFrame
    vix_df = pd.DataFrame([{
        'date': pd.to_datetime(agg.timestamp, unit='ms'),
        'vix_close': agg.close,
        'vix_high': agg.high
    } for agg in aggs])

    return vix_df


def merge_dataframes(df1, df2):
    df1['date'] = pd.to_datetime(df1['date']).dt.normalize()
    df2['date'] = pd.to_datetime(df2['date']).dt.normalize()

    df1 = df1.sort_values('date')
    df2 = df2.sort_values('date')

    merged_df = pd.merge_asof(
        df1,
        df2,
        on='date',
        direction='forward',
        tolerance=pd.Timedelta('3D')
    )

    merged_df = merged_df.rename(columns={'vix_close': 'vix_target'})
    merged_df = merged_df.dropna(subset=['vix_target'])

    return merged_df


def micro_analysis(path, summary_col, headline_col):
    df = load_data(path)
    df, min_date, max_date = sort_by_date(df) #implies dataset is clean, no random dates!!
    df = calculate_sentiment(df, summary_col, pipe, suffix='_summary')
    df = calculate_sentiment(df, headline_col, pipe, suffix='_headline')
    df = calculate_sentiment_stats(df, min_date, max_date, summary_col, headline_col)
    vix_df = download_vix_data(min_date, max_date)
    df = merge_dataframes(df, vix_df)
    return df

def macro_analysis(path, summary_col, headline_col, vix_api_key):
    print("Step 1: Loading data")
    df = load_macro_df(path)
    print(f"Columns after loading: {df.columns.tolist()}")
    print(f"Data shape after loading: {df.shape}")
    
    print("Step 2: Sorting by date")
    if 'date' not in df.columns:
        print(f"ERROR: 'date' column missing before sort_by_date! Available columns: {df.columns.tolist()}")
        # Add a fallback approach
        if 'published_at' in df.columns:
            print("Trying to create date from published_at")
            df['date'] = pd.to_datetime(df['published_at']).dt.date
    
    df, min_date, max_date = sort_by_date(df)
    print(f"Min date: {min_date}, Max date: {max_date}")
    print(f"Columns after sorting: {df.columns.tolist()}")
    
    print("Step 3: Calculating summary sentiment")
    df = calculate_sentiment(df, summary_col, pipe, suffix='_summary')
    print(f"Columns after summary sentiment: {df.columns.tolist()}")
    
    print("Step 4: Calculating headline sentiment")
    df = calculate_sentiment(df, headline_col, pipe, suffix='_headline')
    print(f"Columns after headline sentiment: {df.columns.tolist()}")
    
    print("Step 5: Calculating sentiment stats")
    df = calculate_sentiment_stats(df, min_date, max_date, summary_col, headline_col)
    print(f"Columns after sentiment stats: {df.columns.tolist()}")
    
    print("Step 6: Downloading VIX data")
    vix_df = download_vix_data(min_date, max_date, vix_api_key)
    print(f"VIX data shape: {vix_df.shape}")
    print(f"VIX columns: {vix_df.columns.tolist()}")
    
    print("Step 7: Merging dataframes")
    if 'date' not in df.columns:
        print(f"ERROR: 'date' column missing before merge! Available columns: {df.columns.tolist()}")
        return df  # Return early to see what happened
        
    df = merge_dataframes(df, vix_df)
    print(f"Final data shape: {df.shape}")
    return df
    