def analyze_sentiment(df, col_name, pipe, suffix=''):
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