from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor




def calculate_stats(df, threshold_abs, window=30, window_prev=1):
    df['mean_sentiment_decay'] = df['overall_mean'].shift(1)  # Decay from yesterday
    df['shock_pos'] = df['overall_max'] #best new
    df['shock_neg'] = df['overall_min'] #worst new 
    df['sentiment_dominance'] = df['summary_pos_count'] - df['summary_neg_count'] # which type of new is more prevalent (p/n)?
    df['dominance_trigger'] = (df['sentiment_dominance'] < -5).astype(int) #if one type of news is more prevalent than the other, it will trigger a shock
    df['vix_7d_pct'] = df['vix_target'].pct_change(15)
    df['vix_zscore'] = (df['vix_target'] - df['vix_target'].rolling(30).mean()) / df['vix_target'].rolling(30).std()
    df['ewm_mean'] = df['vix_target'].ewm(span=window).mean()
    df['ewm_std']  = df['vix_target'].ewm(span=window).std()
    df['vix_ewm_zscore'] = (df['vix_target'] - df['ewm_mean']) / df['ewm_std']
    df['mean_prev'] = (
        df['vix_target']
        .rolling(window=window_prev, min_periods=1)
        .mean()
        .shift(1)
    )

    mask = (df['vix_target'] - df['mean_prev']).abs() >= threshold_abs
    df['vix_flat_abs'] = df['vix_target'].where(mask, df['mean_prev'])
    df['ewm_mean_flat'] = df['vix_flat_abs'].ewm(span=window).mean()
    df['ewm_std_flat']  = df['vix_flat_abs'].ewm(span=window).std()
    df['vix_ewm_zscore_flat'] = (df['vix_flat_abs'] - df['ewm_mean_flat']) / df['ewm_std_flat']

    return df

def regime_t_markov_model(X, y, df, k, percentile):
    model = MarkovRegression(y, exog=X, k_regimes=k, switching_variance=False)
    model_results = model.fit(disp=False)

    transition_matrix = np.zeros((k, k))

    for i in range(k):
        row_sum = 0
        for j in range(k):
            if j == k - 1:
            # Last column is implicit: 1 - sum of previous probs
                transition_matrix[i, j] = 1 - row_sum
            else:
                param_name = f'p[{i}->{j}]'
                transition_matrix[i, j] = model_results.params[param_name]
                row_sum += transition_matrix[i, j]


    # Select the most probable regime per time step
    if k == 2:
        df['regime_t_raw'] = model_results.filtered_marginal_probabilities[1]
        df['regime_t'] = df['regime_t_raw'].apply(lambda p: 1 if p >= percentile else 0)
        df['regime_t+1'] = df['regime_t'].shift(-1)
        df['regime_t+1_raw'] = df['regime_t_raw'].shift(-1)
        return transition_matrix, model_results, df
    
    if k == 3:
        df['regime_t_raw'] = model_results.filtered_marginal_probabilities.idxmax(axis=1)
        df['regime_t'] = df['regime_t_raw'].apply(lambda r: 1 if r in [1, 2] else 0)

        # Optional: store raw probabilities for analysis
        df['regime_0_prob'] = model_results.filtered_marginal_probabilities[0]
        df['regime_1_prob'] = model_results.filtered_marginal_probabilities[1]
        df['regime_2_prob'] = model_results.filtered_marginal_probabilities[2]

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

def identify_sustained_regimes_and_transitions(classification_df, min_duration=1, trade=False):
    """
    Identifies regime 1 blocks and transitions in the classification dataframe.

    Args:
    classification_df (pd.DataFrame): DataFrame containing 'regime_t' column (regime states).
    min_duration (int): Minimum duration (in days) for a sustained regime 1 block (default is 7).
    trade (bool): If True, ignore min_duration for sustained regimes.

    Returns:
    pd.DataFrame: DataFrame with new columns indicating sustained regimes and transitions.
    """
    # If trade mode, no minimum duration filter
    if trade:
        min_duration = 0

    # Step 1: Identify regime groups
    classification_df['regime_group'] = (classification_df['regime_t'] != classification_df['regime_t'].shift()).cumsum()

    regime_lengths = classification_df.groupby('regime_group')['regime_t'].agg(['first', 'size'])

    # Valid sustained regimes based on min_duration
    valid_groups = regime_lengths[(regime_lengths['first'] == 1) & (regime_lengths['size'] >= min_duration)].index

    classification_df['is_sustained_regime1'] = classification_df['regime_group'].isin(valid_groups)

    # Step 2: Identify **all** transitions from 0 → 1 (ignore duration for this)
    classification_df['regime_t-1'] = classification_df['regime_t'].shift(1)

    classification_df['is_transition'] = (
        (classification_df['regime_t-1'] == 0) &
        (classification_df['regime_t'] == 1)
    )

    return classification_df


def transform_vix_data(classification_df,k, window=60, trade=False, test=False):
    """
    Transforms the VIX data into windows around every regime switch from 0→1 or 0→2.
    - trade=False: returns past & future windows around each switch
    - trade=True: returns only past windows (for live trading)
    - test=True: pads short windows with NaN instead of skipping
    """
    df = classification_df.copy()
    df['vix_target_t+1'] = df['vix_target'].shift(-1)

    # 1) detect every 0→1 or 0→2 switch in regime_t_raw
    if k == 2:
        df['prev_regime'] = df['regime_t'].shift(1).fillna(0).astype(int)
        transition_idxs = df.index[
            (df['prev_regime'] == 0) &
            (df['regime_t'] == 1)
        ]
    elif k == 3:
        transition_idxs = df.index[
            (df['prev_regime'] == 0) &
            (df['regime_t'].isin([1,2]))
        ]

    if not trade:
        vix_windows_train      = []
        regime_windows_train   = []
        vix_windows_past_train = []
        transition_times       = []

        for idx in transition_idxs:
            loc = df.index.get_loc(idx)

            # — past window —
            start_past = loc - window
            if start_past < 0:
                if test:
                    past_vals = df.iloc[:loc]['vix_target'].values
                    pad       = np.full(window - past_vals.size, np.nan)
                    v_past    = np.concatenate([pad, past_vals])
                else:
                    continue
            else:
                v_past = df.iloc[start_past:loc]['vix_target'].values

            # — future window —
            end_fut = loc + window
            if end_fut > len(df):
                if test:
                    fut_vals = df.iloc[loc:end_fut]['vix_target_t+1'].values
                    pad      = np.full(window - fut_vals.size, np.nan)
                    v_fut    = np.concatenate([fut_vals, pad])

                    reg_vals = df.iloc[loc:end_fut]['regime_t'].values
                    pad_r    = np.full(window - reg_vals.size, np.nan)
                    r_fut    = np.concatenate([reg_vals, pad_r])
                else:
                    continue
            else:
                v_fut = df.iloc[loc:end_fut]['vix_target_t+1'].values
                r_fut = df.iloc[loc:end_fut]['regime_t'].values

            vix_windows_train.append(v_fut)
            regime_windows_train.append(r_fut)
            vix_windows_past_train.append(v_past)
            transition_times.append(idx)

        dt_index = pd.to_datetime(transition_times)
        cols     = [f"Day {i}" for i in range(1, window+1)]

        vix_windows_df_train      = pd.DataFrame(vix_windows_train,      index=dt_index, columns=cols)
        regime_windows_df_train   = pd.DataFrame(regime_windows_train,   index=dt_index, columns=cols)
        vix_windows_past_df_train = pd.DataFrame(vix_windows_past_train, index=dt_index, columns=cols)

        for df_ in (vix_windows_df_train, regime_windows_df_train, vix_windows_past_df_train):
            df_.index.name = "transition_time"

        return {
            "vix_windows_df_train":      vix_windows_df_train,
            "regime_windows_df_train":   regime_windows_df_train,
            "vix_windows_past_df_train": vix_windows_past_df_train
        }

    else:
        # trade mode: only past windows
        vix_windows_past_train = []
        transition_times       = []

        for idx in transition_idxs:
            loc = df.index.get_loc(idx)
            start_past = loc - window

            if start_past < 0:
                if test:
                    past_vals = df.iloc[:loc]['vix_target'].values
                    pad       = np.full(window - past_vals.size, np.nan)
                    v_past    = np.concatenate([pad, past_vals])
                else:
                    continue
            else:
                v_past = df.iloc[start_past:loc]['vix_target'].values

            vix_windows_past_train.append(v_past)
            transition_times.append(idx)

        dt_index = pd.to_datetime(transition_times)
        cols     = [f"Day {i}" for i in range(1, window+1)]

        vix_windows_past_df_train = pd.DataFrame(vix_windows_past_train,
                                                 index=dt_index,
                                                 columns=cols)
        vix_windows_past_df_train.index.name = "transition_time"

        return { "vix_windows_past_df_train": vix_windows_past_df_train }



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
    vix_early = df.iloc[:, :25]
    vix_late  = df.iloc[:, 35:]
    vix_start = df.iloc[:, 0]

    early_spike = (vix_early.max(axis=1) > vix_start * 1.2)
    late_spike  = (vix_late.max(axis=1) > vix_start * 1.2)

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
    return df, clf1, vix_pct_change_lookback_df


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


def detect_spike_arc_within_regime(vix_path, regime_path,
                                   search_back=15, min_distance=8):
    import numpy as np

    v = np.asarray(vix_path, dtype=np.float32)
    r = np.asarray(regime_path, dtype=np.int8)

    # 1) locate the very first regime==1 day
    ones = np.where(r == 1)[0]
    if len(ones) == 0:
        return 0, 0
    block_start = ones[0]

    # 2) extend forward to cover the full contiguous block of 1’s
    block_end = block_start
    while block_end + 1 < len(r) and r[block_end+1] == 1:
        block_end += 1

    # now our block is [block_start … block_end]
    idxs = np.arange(block_start, block_end+1)
    v_block = v[idxs]

    # 3) pick the true max inside that block
    peak_rel = int(np.argmax(v_block))
    peak_idx = idxs[peak_rel]

    # if that max is at the very start of the block, look for next highest
    if peak_rel == 0:
        sorted_rel = np.argsort(v_block)[::-1]
        for rel in sorted_rel[1:]:
            if rel > 0:
                peak_rel = int(rel)
                peak_idx = idxs[peak_rel]
                break
        else:
            return block_start, block_start

    # 4) local trough: only search between
    #    [ max(block_start, peak_idx-search_back)  …  peak_idx ]
    win0 = max(block_start, peak_idx - search_back)
    pre = v[win0:peak_idx]
    if len(pre) == 0:
        return block_start, peak_idx

    trough_rel = int(np.argmin(pre))
    trough_idx = win0 + trough_rel

    # 5) enforce a minimum rise distance
    if (peak_idx - trough_idx) < min_distance:
        trough_idx = block_start

    return trough_idx, peak_idx


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
