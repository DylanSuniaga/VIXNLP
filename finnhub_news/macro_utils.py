from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor




def calculate_stats(df):
    df['mean_sentiment_decay'] = df['overall_mean'].shift(1)  # Decay from yesterday
    df['shock_pos'] = df['overall_max'] #best new
    df['shock_neg'] = df['overall_min'] #worst new 
    df['sentiment_dominance'] = df['summary_pos_count'] - df['summary_neg_count'] # which type of new is more prevalent (p/n)?
    df['dominance_trigger'] = (df['sentiment_dominance'] < -5).astype(int) #if one type of news is more prevalent than the other, it will trigger a shock
    df['vix_7d_pct'] = df['vix_target'].pct_change(15)
    df['vix_zscore'] = (df['vix_target'] - df['vix_target'].rolling(30).mean()) / df['vix_target'].rolling(30).std()
    df = df.dropna()
    return df

def regime_t_markov_model(X, y, df):
    model = MarkovRegression(y, exog=X, k_regimes=2, switching_variance=False)
    model_results = model.fit(disp=False)

    P_00 = model_results.params['p[0->0]']       # P(stay in regime 0)
    P_10 = model_results.params['p[1->0]']       # P(switch from regime 1 to 0)
    P_01 = 1 - P_00                       # P(switch from regime 0 to 1)
    P_11 = 1 - P_10                       # P(stay in regime 1)

    transition_matrix = np.array([
        [P_00, P_01],  # from regime 0
        [P_10, P_11]   # from regime 1
    ])

    df['regime_t'] = model_results.filtered_marginal_probabilities[1].apply(lambda p: 1 if p >= 0.6 else 0)
    df['regime_t_raw'] = model_results.filtered_marginal_probabilities[1]
    df['regime_t+1'] = df['regime_t'].shift(-1)
    df['regime_t+1_raw'] = df['regime_t_raw'].shift(-1)
    df = df.dropna()
    return transition_matrix, model_results, df


def clf_panic_tomorrow(df, features, target):
    X_cls = df[features]
    y_cls = df[target]
    clf = RandomForestClassifier(n_estimators=400, max_depth=7, random_state=42)
    clf.fit(X_cls, y_cls)

# Evaluate
    y_pred = clf.predict(X_cls)
    print(classification_report(y_cls, y_pred))

    probs = clf.predict_proba(X_cls)[:, 1]
    df['panic_prob'] = probs

    df.index = pd.to_datetime(df.index)

    return clf, y_pred, df

def identify_sustained_regimes_and_transitions(classification_df, min_duration=5):
    """
    Identifies sustained regime 1 blocks and transitions in the classification dataframe.

    Args:
    classification_df (pd.DataFrame): DataFrame containing 'regime_t' column (regime states).
    min_duration (int): Minimum duration (in days) for a regime 1 block to be considered valid (default is 7).

    Returns:
    pd.DataFrame: DataFrame with new columns indicating sustained regime 1 and transitions.
    """
    # Step 1: Identify sustained regime 1 blocks (min_duration + consecutive days)
    classification_df['regime_group'] = (classification_df['regime_t'] != classification_df['regime_t'].shift()).cumsum()

    # Group by consecutive regime IDs
    regime_lengths = classification_df.groupby('regime_group')['regime_t'].agg(['first', 'size'])
    
    # Identify valid groups based on 'regime_t' being 1 and duration being >= min_duration
    valid_groups = regime_lengths[(regime_lengths['first'] == 1) & (regime_lengths['size'] >= min_duration)].index

    # Step 2: Only retain regime 1 rows that belong to valid sustained panic groups
    classification_df['is_sustained_regime1'] = classification_df['regime_group'].isin(valid_groups)

    # Step 3: Redefine transitions: 0 → 1 *and* part of a valid (sustained) panic block
    classification_df['regime_t-1'] = classification_df['regime_t'].shift(1)
    classification_df['is_transition'] = (
        (classification_df['regime_t-1'] == 0) &
        (classification_df['regime_t'] == 1) &
        (classification_df['is_sustained_regime1'])
    )

    return classification_df

def transform_vix_data(classification_df, window=60):
    """
    Transforms the VIX data into three separate DataFrames for training:
    - VIX windows (future values)
    - Regime windows (future regime states)
    - VIX windows (past values)

    Args:
    classification_df (pd.DataFrame): DataFrame containing the raw classification data with VIX and regimes.
    window (int): The size of the window for future and past data.

    Returns:
    dict: A dictionary containing the transformed DataFrames.
    """
    vix_windows_train = []
    regime_windows_train = []
    vix_windows_past_train = []  # For lookback VIX
    transition_datetimes = [] 
    classification_df['vix_target_t+1'] = classification_df['vix_target'].shift(-1)

    # Iterate through the classification_df to extract the relevant windows
    for idx in classification_df[classification_df['is_transition']].index:
        loc = classification_df.index.get_loc(idx)
        
        # Ensure both future and past slices are valid length
        if loc - window < 0 or loc + window > len(classification_df):
            continue

        # Look-ahead VIX + regime
        vix_slice_future = classification_df.iloc[loc:loc + window]["vix_target_t+1"].values
        regime_slice_future = classification_df.iloc[loc:loc + window]["regime_t_raw"].values

        # Lookback VIX (before the transition point)
        vix_slice_past = classification_df.iloc[loc - window:loc]["vix_target"].values

        # Only store if all are full length
        if len(vix_slice_future) == window and len(vix_slice_past) == window:
            vix_windows_train.append(vix_slice_future)
            regime_windows_train.append(regime_slice_future)
            vix_windows_past_train.append(vix_slice_past)
            transition_datetimes.append(idx)

    # Use datetime index for all 3 dataframes
    dt_index = pd.to_datetime(transition_datetimes)

    # Create DataFrames
    vix_windows_df_train = pd.DataFrame(vix_windows_train, index=dt_index)
    regime_windows_df_train = pd.DataFrame(regime_windows_train, index=dt_index)
    vix_windows_past_df_train = pd.DataFrame(vix_windows_past_train, index=dt_index)

    # Set index and column names
    for df in [vix_windows_df_train, regime_windows_df_train, vix_windows_past_df_train]:
        df.index.name = "transition_time"
        df.columns = [f"Day {i}" for i in range(1, window + 1)]

    # Return the three DataFrames
    return {
        "vix_windows_df_train": vix_windows_df_train,
        "regime_windows_df_train": regime_windows_df_train,
        "vix_windows_past_df_train": vix_windows_past_df_train
    }


import numpy as np
import pandas as pd

def calculate_pct_changes(df_model, valid_indices, lookback=30, vix_column="vix_target"):
    """
    This function calculates the percentage change in VIX for the given lookback period for valid indices.

    Args:
        df_model (pd.DataFrame): The DataFrame containing VIX data and other features.
        valid_indices (list): List of valid indices for which to calculate percentage change.
        lookback (int): The lookback period for calculating the percentage change.
        vix_column (str): The name of the column containing the VIX values.

    Returns:
        pd.DataFrame: DataFrame of calculated percentage changes with a specified lookback period.
    """
    pct_change_series = []
    used_indices = []

    for idx in valid_indices:
        if idx not in df_model.index:
            continue

        loc = df_model.index.get_loc(idx)
        if loc - lookback - 1 < 0:
            continue  # need one extra point for lookback pct changes

        # Get values for the specified lookback period (including current day)
        vix_window = df_model.iloc[loc - lookback - 1:loc][vix_column]

        if vix_window.isna().any() or len(vix_window) != lookback + 1:
            continue

        # Calculate percentage change over the lookback period
        vix_pct_changes = vix_window.pct_change().iloc[1:].values  # Exclude the first NaN value
        if len(vix_pct_changes) == lookback:
            pct_change_series.append(vix_pct_changes)
            used_indices.append(idx)

    # Final DataFrame (shape: n_transitions × lookback)
    pct_change_df = pd.DataFrame(
        pct_change_series,
        index=pd.to_datetime(used_indices),
        columns=[f"Day -{i}" for i in range(lookback, 0, -1)]
    )

    return pct_change_df, used_indices


# binary classifier for delayed vs immediate spikes
def clf_delayed_spike_prob(df, df1, target): #df should be vix_windows_df_train
    vix_early = df.iloc[:, :30]
    vix_late  = df.iloc[:, 30:]
    vix_start = df.iloc[:, 0]

    early_spike = (vix_early.max(axis=1) > vix_start * 1.1)
    late_spike  = (vix_late.max(axis=1) > vix_start * 1.1)

# Final label: 1 = spike happens early, 0 = spike only late, NaN = no spike
    spike_label = pd.Series(np.where(early_spike, 1, np.where(late_spike, 0, np.nan)), index=df.index)
    spike_label = spike_label.dropna()
    valid_indices = spike_label.index

    vix_pct_change_lookback_df, used_indices = calculate_pct_changes(df1, valid_indices, lookback=30, vix_column=target)
    clf1 = RandomForestClassifier(n_estimators=400, max_depth=7, random_state=42)
    y = spike_label.loc[used_indices].astype(int)
    clf1.fit(vix_pct_change_lookback_df, y)

    if len(vix_pct_change_lookback_df) > 1:
        early_spike_probs = clf1.predict_proba(vix_pct_change_lookback_df)[:, 1]
        df.loc[used_indices, "early_spike_prob"] = early_spike_probs

        early_spike_probs = clf1.predict_proba(vix_pct_change_lookback_df)[:, 1]
        df.loc[vix_pct_change_lookback_df.index, "early_spike_prob"] = early_spike_probs
    else:
        early_spike_probs = clf1.predict_proba(vix_pct_change_lookback_df)
        df.loc[vix_pct_change_lookback_df.index, "early_spike_prob"] = early_spike_probs
        early_spike_probs = clf1.predict_proba(vix_pct_change_lookback_df)
        df.loc[vix_pct_change_lookback_df.index, "early_spike_prob"] = early_spike_probs


    #y_true = spike_label.loc[df.index].astype(int)
    #y_pred = (early_spike_probs > 0.5).astype(int)
    return df

def detect_spike_arc(vix_path, search_back=15, min_distance=8):
    vix_path = np.asarray(vix_path, dtype=np.float32)
    peak_idx = int(np.argmax(vix_path))

    if peak_idx == 0:
        sorted_peaks = np.argsort(vix_path)[::-1]
        for alt_peak in sorted_peaks[1:]:
            if alt_peak > 5:
                peak_idx = alt_peak
                break
        else:
            return 0, 0

    window_start = max(0, peak_idx - search_back)
    pre_peak = vix_path[window_start:peak_idx]

    if len(pre_peak) == 0:
        return 0, peak_idx

    local_min_idx = int(np.argmin(pre_peak)) + window_start
    if (peak_idx - local_min_idx) < min_distance:
        global_min_idx = int(np.argmin(vix_path[:peak_idx])) if peak_idx > 0 else 0
        return global_min_idx, peak_idx

    return local_min_idx, peak_idx

import numpy as np
from sklearn.linear_model import LinearRegression

def linear_reg_models(vix_windows_df_train, min_length=1):
    """
    Splits each 60‑day VIX path into its rising arc, buckets them into
    'long' (>10 days) and 'sharp' (<=10 days) rises, then fits a
    LinearRegression on each group if there’s >1 sample.

    Returns:
      model_long, model_sharp, 
      y_pred_long, y_pred_sharp,
      sharp_rises_X, long_rises_X,
      sharp_rises_y, long_rises_y,
      long_Xc, long_yc, sharp_Xc, sharp_yc
    """

    long_rises_X, long_rises_y = [], []
    sharp_rises_X, sharp_rises_y = [], []

    # 1) extract each row's numeric 60‑day array and slice its rising arc
    for _, row in vix_windows_df_train.iterrows():
        # pull columns "Day 1".."Day 60" as floats
        vix_path = row.filter(like="Day ").to_numpy(dtype=float)
        start_idx, peak_idx = detect_spike_arc(vix_path)

        # slice the actual values
        rising = vix_path[start_idx : peak_idx + 1]
        if len(rising) < min_length:
            continue

        # build time index for regression
        t = np.arange(len(rising)).reshape(-1, 1)

        if len(rising) > 10:
            long_rises_X.append(t)
            long_rises_y.append(rising.reshape(-1, 1))
        else:
            sharp_rises_X.append(t)
            sharp_rises_y.append(rising.reshape(-1, 1))

    # 2) helper to stack & fit if possible
    def _stack_and_fit(X_list, y_list):
        if not X_list:
            return None, None, None, None
        Xc = np.vstack(X_list)
        yc = np.vstack(y_list)
        if Xc.shape[0] > 1:
            model = MLPRegressor(hidden_layer_sizes=(10000, 5000, 2500, 1000, 500, 250, 100, 5), activation='relu', max_iter=1000)
            model.fit(Xc, yc.ravel())  # Flatten y
            y_pred = model.predict(Xc)
            return model, Xc, yc, y_pred
        else:
            return None, Xc, yc, None

    model_long,  long_Xc,  long_yc,  y_pred_long  = _stack_and_fit(long_rises_X,  long_rises_y)
    model_sharp, sharp_Xc, sharp_yc, y_pred_sharp = _stack_and_fit(sharp_rises_X, sharp_rises_y)

    return (
        model_long, model_sharp,
        y_pred_long, y_pred_sharp,
        sharp_rises_X, long_rises_X,
        sharp_rises_y, long_rises_y,
        long_Xc, long_yc,
        sharp_Xc, sharp_yc
    )
