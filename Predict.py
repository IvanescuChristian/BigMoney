import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta
import os
import glob
import warnings

warnings.filterwarnings('ignore')

INPUT_DIR = os.path.join(os.getcwd(), 'historical_hourly')
OUTPUT_DIR = os.path.join(os.getcwd(), 'predicted_hourly')

HORIZON = 168
NUM_SIMULATIONS = 50

PRICE_FEATURES = ['total_volume_usd', 'social_volume', 'hourly_increase', 'price-pop',
                   'price-soc correlation', 'RSI', 'volatility']

SOCIAL_FEATURES = ['social_lag_1d', 'social_lag_7d', 'social_avg_7d',
                    'price_pct_1d', 'volume_avg', 'day_of_week']

MAGNITUDE_FEATURES = ['volatility', 'RSI', 'social_volume', 'total_volume_usd',
                       'abs_change_lag1', 'abs_change_lag24', 'hour', 'day_of_week',
                       'whale_pressure', 'whale_activity']

WHALE_DIR = os.path.join(os.getcwd(), 'whale_data')
WHALE_SCORE_FILE = os.path.join(WHALE_DIR, 'daily_scores.csv')

# Wrapped/staked tokens -> parent coin + expected ratio range
# ratio = wrapped_price / parent_price
# If computed ratio falls outside (min_ratio, max_ratio), use median from data
WRAPPED_TOKENS = {
    'staked-ether':         {'parent': 'ethereum',  'min_ratio': 0.95, 'max_ratio': 1.05},
    'wrapped-steth':        {'parent': 'ethereum',  'min_ratio': 0.80, 'max_ratio': 1.25},
    'weth':                 {'parent': 'ethereum',  'min_ratio': 0.99, 'max_ratio': 1.01},
    'coinbase-wrapped-btc': {'parent': 'bitcoin',   'min_ratio': 0.95, 'max_ratio': 1.05},
    'jito-staked-sol':      {'parent': 'solana',    'min_ratio': 1.00, 'max_ratio': 1.50},
    'wrapped-eeth':         {'parent': 'ethereum',  'min_ratio': 0.95, 'max_ratio': 1.15},
}


def load_all_historical_data():
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not all_files:
        return None

    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            sv_col = [c for c in temp_df.columns if 'social_volume' in c]
            if sv_col:
                temp_df['social_volume'] = temp_df[sv_col[0]]
            else:
                temp_df['social_volume'] = 0.0

            needed = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd',
                      'social_volume', 'hourly_increase', 'price-soc correlation',
                      'price-pop', 'RSI', 'volatility']
            for col in needed:
                if col not in temp_df.columns:
                    temp_df[col] = 0.0
            df_list.append(temp_df[needed])
        except:
            pass

    if not df_list:
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp_utc'] = pd.to_datetime(full_df['timestamp_utc'])
    full_df = full_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Merge whale data (daily scores joined by date)
    full_df['date_key'] = full_df['timestamp_utc'].dt.strftime('%Y-%m-%d')
    if os.path.exists(WHALE_SCORE_FILE):
        whale_df = pd.read_csv(WHALE_SCORE_FILE)
        whale_cols = ['date', 'whale_pressure', 'whale_activity']
        whale_cols = [c for c in whale_cols if c in whale_df.columns]
        if 'date' in whale_cols:
            whale_df = whale_df[whale_cols].rename(columns={'date': 'date_key'})
            full_df = full_df.merge(whale_df, on='date_key', how='left')
            filled = full_df['whale_pressure'].notna().sum()
            print(f"    -> Whale data merged: {filled}/{len(full_df)} rows matched")
    for wc in ['whale_pressure', 'whale_activity']:
        if wc not in full_df.columns:
            full_df[wc] = 0.0
    full_df = full_df.drop(columns=['date_key']).fillna(0.0)

    print(f"    -> {len(all_files)} fisiere zilnice. Total: {len(full_df)} randuri.")
    return full_df.sort_values(['coin_id', 'timestamp_utc'])


# -- dynamic indicators for Monte Carlo --

def get_dynamic_rsi(prices_matrix, period=14):
    if prices_matrix.shape[1] < period + 1:
        return np.full(prices_matrix.shape[0], 50.0)
    deltas = np.diff(prices_matrix[:, -period - 1:], axis=1)
    ups = np.sum(np.where(deltas > 0, deltas, 0), axis=1) / period
    downs = np.sum(np.where(deltas < 0, -deltas, 0), axis=1) / period
    downs = np.where(downs == 0, 1e-9, downs)
    rsi = 100.0 - (100.0 / (1.0 + (ups / downs)))
    return np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)


def get_dynamic_volat(prices_matrix, period=24):
    if prices_matrix.shape[1] < period:
        return np.zeros(prices_matrix.shape[0])
    arr = prices_matrix[:, -period:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(arr, axis=1) / (arr[:, :-1] + 1e-9)
    return np.nan_to_num(np.std(rets, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================================
#  STAGE 1: Social Volume — holdout-validated bias correction
# =========================================================================

def train_social_model(c_df):
    """Train daily social model with temporal train/test split for bias measurement."""
    sdf = c_df.copy()
    sdf['date'] = sdf['timestamp_utc'].dt.date

    daily = sdf.groupby('date').agg(
        social_volume=('social_volume', 'first'),
        price_start=('price_usd', 'first'),
        price_end=('price_usd', 'last'),
        volume_avg=('total_volume_usd', 'mean')
    ).reset_index().sort_values('date')

    daily['social_lag_1d'] = daily['social_volume'].shift(1)
    daily['social_lag_7d'] = daily['social_volume'].shift(7)
    daily['social_avg_7d'] = daily['social_volume'].rolling(7, min_periods=1).mean()
    daily['price_pct_1d'] = daily['price_end'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    daily['day_of_week'] = pd.to_datetime(daily['date']).dt.dayofweek
    daily = daily.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if len(daily) < 14:
        return None, daily, 1.0

    # Temporal split: train on first 80%, measure bias on last 20%
    split_idx = int(len(daily) * 0.8)
    train_daily = daily.iloc[:split_idx]
    holdout_daily = daily.iloc[split_idx:]

    X_train = train_daily[SOCIAL_FEATURES]
    y_train = train_daily['social_volume']
    X_holdout = holdout_daily[SOCIAL_FEATURES]
    y_holdout = holdout_daily['social_volume']

    model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        reg_alpha=1.0, reg_lambda=2.0, n_jobs=-1
    )
    try:
        model.fit(X_train, y_train)

        # Measure bias on HOLDOUT (not training data!)
        holdout_preds = model.predict(X_holdout)
        actual_mean = float(y_holdout.mean()) if y_holdout.mean() > 0 else 0.01
        pred_mean = float(holdout_preds.mean()) if holdout_preds.mean() > 0 else 0.01
        bias_ratio = pred_mean / actual_mean

        return model, daily, bias_ratio
    except:
        return None, daily, 1.0


def predict_social_series(social_result, c_df, horizon=HORIZON):
    """Predict daily social with holdout-measured bias correction."""
    if social_result is None:
        last_soc = max(float(c_df['social_volume'].iloc[-1]), 0.0)
        return [last_soc] * horizon

    social_model, daily, bias_ratio = social_result

    # If no model or extreme bias, fallback to rolling median
    if social_model is None or bias_ratio > 3.0 or bias_ratio < 0.3:
        median_soc = max(float(daily['social_volume'].tail(14).median()), 0.0)
        return [median_soc] * horizon

    # Correction: scale predictions down by measured bias
    correction = 1.0 / max(bias_ratio, 0.5)
    correction = min(correction, 2.0)  # don't over-correct upward either

    hist_median = float(daily['social_volume'].median())
    hist_p90 = float(daily['social_volume'].quantile(0.9))
    clamp_max = max(hist_p90 * 1.3, hist_median * 2, 1.0)

    last_ts = c_df['timestamp_utc'].iloc[-1]
    social_hist = list(daily['social_volume'].tail(7).values)
    last_vol_avg = float(daily['volume_avg'].iloc[-1])
    last_price_pct = float(daily['price_pct_1d'].iloc[-1])

    num_days = (horizon + 23) // 24
    daily_preds = []

    for d in range(num_days):
        t_day = last_ts + timedelta(days=d + 1)
        lag_1d = social_hist[-1]
        lag_7d = social_hist[-7] if len(social_hist) >= 7 else social_hist[0]
        avg_7d = float(np.mean(social_hist[-7:]))

        inp = pd.DataFrame([{
            'social_lag_1d': lag_1d,
            'social_lag_7d': lag_7d,
            'social_avg_7d': avg_7d,
            'price_pct_1d': last_price_pct,
            'volume_avg': last_vol_avg,
            'day_of_week': t_day.weekday()
        }])

        pred_soc = float(social_model.predict(inp)[0])
        pred_soc = max(pred_soc * correction, 0.0)
        pred_soc = min(pred_soc, clamp_max)
        daily_preds.append(pred_soc)
        social_hist.append(pred_soc)

    hourly_preds = []
    for dv in daily_preds:
        hourly_preds.extend([dv] * 24)
    return hourly_preds[:horizon]


# =========================================================================
#  STAGE 2: Magnitude Prediction
# =========================================================================

def train_magnitude_model(c_df):
    mdf = c_df.copy()
    mdf['pct_change'] = mdf['price_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    mdf['abs_pct_change'] = mdf['pct_change'].abs()
    mdf['abs_change_lag1'] = mdf['abs_pct_change'].shift(1)
    mdf['abs_change_lag24'] = mdf['abs_pct_change'].shift(24)
    mdf['hour'] = mdf['timestamp_utc'].dt.hour
    mdf['day_of_week'] = mdf['timestamp_utc'].dt.dayofweek
    mdf = mdf.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if len(mdf) < 50:
        return None, 0.001

    X = mdf[MAGNITUDE_FEATURES]
    y = mdf['abs_pct_change']

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, n_jobs=-1)
    try:
        model.fit(X, y)
        return model, float(y.mean())
    except:
        return None, float(y.mean()) if len(y) > 0 else 0.001


def predict_magnitude_series(mag_model, c_df, social_preds, avg_mag, horizon=HORIZON):
    if mag_model is None:
        return [avg_mag] * horizon

    last_vol = c_df['total_volume_usd'].iloc[-1]
    last_rsi = c_df['RSI'].iloc[-1]
    last_volat = c_df['volatility'].iloc[-1]
    last_ts = c_df['timestamp_utc'].iloc[-1]

    # Whale features (use last known values)
    last_whale_pressure = float(c_df['whale_pressure'].iloc[-1]) if 'whale_pressure' in c_df.columns else 0.0
    last_whale_activity = float(c_df['whale_activity'].iloc[-1]) if 'whale_activity' in c_df.columns else 0.0

    pct_changes = c_df['price_usd'].pct_change().abs().fillna(0)
    mag_hist = list(pct_changes.tail(24).values)

    predicted = []
    for h in range(horizon):
        t_curr = last_ts + timedelta(hours=h + 1)
        lag1 = mag_hist[-1]
        lag24 = mag_hist[-24] if len(mag_hist) >= 24 else mag_hist[0]

        inp = pd.DataFrame([{
            'volatility': last_volat, 'RSI': last_rsi,
            'social_volume': social_preds[h], 'total_volume_usd': last_vol,
            'abs_change_lag1': lag1, 'abs_change_lag24': lag24,
            'hour': t_curr.hour, 'day_of_week': t_curr.dayofweek,
            'whale_pressure': last_whale_pressure,
            'whale_activity': last_whale_activity
        }])

        pred_mag = max(float(mag_model.predict(inp)[0]), 0.0005)
        predicted.append(pred_mag)
        mag_hist.append(pred_mag)

    return predicted


# =========================================================================
#  STAGE 3: Price Prediction with Monte Carlo
# =========================================================================

def predict_coin(coin, c_df, df_all):
    """Run full 3-stage prediction for one coin."""
    c_df = c_df.copy().sort_values('timestamp_utc')
    c_df['lag_1'] = c_df['price_usd'].shift(1)
    c_df['lag_24'] = c_df['price_usd'].shift(24)

    train_data = c_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if len(train_data) < 50:
        print(f"    ! Skip {coin} -- not enough data ({len(train_data)} rows)")
        return None, None

    # -- Stage 1: Social (holdout bias correction) --
    social_result = train_social_model(c_df)
    social_model_obj = social_result[0] if social_result[0] is not None else None
    bias_ratio = social_result[2] if len(social_result) > 2 else 1.0
    social_preds = predict_social_series(social_result, c_df, HORIZON)

    if social_model_obj is None or bias_ratio > 3.0 or bias_ratio < 0.3:
        soc_tag = "MED"
    else:
        soc_tag = f"b={bias_ratio:.1f}"

    # -- Stage 2: Magnitude --
    mag_model, avg_mag = train_magnitude_model(c_df)
    mag_preds = predict_magnitude_series(mag_model, c_df, social_preds, avg_mag, HORIZON)
    mag_tag = "XGB" if mag_model else "AVG"
    avg_pred_mag = np.mean(mag_preds) * 100
    avg_pred_soc = np.mean(social_preds)

    print(f"    -> {coin:<30} soc={soc_tag}({avg_pred_soc:.1f}) "
          f"mag={mag_tag}({avg_pred_mag:.2f}%) train={len(train_data)}h  ", end="")

    # -- Stage 3: Price --
    X_cols = PRICE_FEATURES + ['lag_1', 'lag_24']
    X = train_data[X_cols]
    y = train_data['price_usd']

    price_model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04,
                                     max_depth=6, n_jobs=-1)
    try:
        price_model.fit(X, y)
    except Exception as e:
        print(f"EROARE XGB: {e}")
        return None, None

    last_row = c_df.iloc[-1]
    t_curr = last_row['timestamp_utc']
    historical_drift = train_data['hourly_increase'].mean()

    base_hist = np.array(c_df['price_usd'].tail(50).tolist())
    price_mat = np.tile(base_hist, (NUM_SIMULATIONS, 1))

    s_vol = last_row['total_volume_usd']
    s_pop = last_row['price-pop']
    s_corr = last_row['price-soc correlation']

    # Social volume baseline for magnitude boost
    social_median = max(float(c_df['social_volume'].median()), 1.0)
    # Whale pressure from last known day
    last_whale_p = float(c_df['whale_pressure'].iloc[-1]) if 'whale_pressure' in c_df.columns else 0.0
    last_whale_a = float(c_df['whale_activity'].iloc[-1]) if 'whale_activity' in c_df.columns else 0.0

    coin_results = []
    coin_predicted_prices = []

    for h in range(HORIZON):
        t_curr += timedelta(hours=1)
        s_soc = social_preds[h]
        expected_magnitude = mag_preds[h]

        # SOCIAL MAGNITUDE BOOST
        social_ratio = s_soc / social_median if social_median > 0 else 1.0
        if social_ratio > 1.5:
            social_mag_boost = 1.0 + 0.4 * np.log2(social_ratio)
            expected_magnitude = expected_magnitude * min(social_mag_boost, 3.0)

        # WHALE ACTIVITY BOOST: high whale activity = bigger expected moves
        # whale_activity > 0.3 = whales are unusually active
        if last_whale_a > 0.3:
            whale_boost = 1.0 + last_whale_a * 0.5  # activity=0.5 -> 1.25x, activity=1.0 -> 1.5x
            expected_magnitude = expected_magnitude * min(whale_boost, 2.0)

        l1 = price_mat[:, -1]
        l24 = price_mat[:, -24] if price_mat.shape[1] >= 24 else price_mat[:, 0]
        inc = l1 - price_mat[:, -2]

        rsi_vec = get_dynamic_rsi(price_mat)
        volat_vec = get_dynamic_volat(price_mat)
        volat_vec = np.maximum(volat_vec, 0.0015)

        inp = pd.DataFrame({
            'total_volume_usd': np.full(NUM_SIMULATIONS, s_vol),
            'social_volume': np.full(NUM_SIMULATIONS, s_soc),
            'hourly_increase': inc,
            'price-pop': np.full(NUM_SIMULATIONS, s_pop),
            'price-soc correlation': np.full(NUM_SIMULATIONS, s_corr),
            'RSI': rsi_vec, 'volatility': volat_vec,
            'lag_1': l1, 'lag_24': l24
        })[X_cols]

        base_preds = price_model.predict(inp)

        drift_array = np.full(NUM_SIMULATIONS, historical_drift)
        capped_mag = min(expected_magnitude, 0.04)  # raised cap to 4% for social spikes
        anchor_price = base_hist[-1]
        noise_scale = capped_mag * anchor_price
        noise = np.random.normal(drift_array, max(noise_scale, 1e-9))
        final_preds = np.maximum(base_preds + noise, 1e-9)
        final_preds = np.clip(final_preds, anchor_price * 0.5, anchor_price * 1.5)

        mean_price = np.mean(final_preds)
        std_error = np.std(final_preds)
        mean_inc = np.mean(final_preds - l1)

        price_mat = np.column_stack((price_mat[:, 1:], final_preds))
        coin_predicted_prices.append(mean_price)

        coin_results.append({
            'timestamp_utc': t_curr, 'coin_id': coin,
            'price_usd': mean_price, 'error': std_error,
            'total_volume_usd': s_vol, 'hourly_increase': mean_inc,
            'price-soc correlation': s_corr, 'price-pop': s_pop,
            'predicted_social_volume': s_soc,
            'predicted_magnitude_pct': expected_magnitude * 100
        })

    start_price = coin_predicted_prices[0]
    end_price = coin_predicted_prices[-1]
    pct_change = ((end_price - start_price) / start_price) * 100

    summary = {
        'coin_id': coin,
        'start_prediction_usd': start_price,
        'end_prediction_usd': end_price,
        'predicted_change_%': round(pct_change, 2),
        'avg_predicted_social': round(avg_pred_soc, 2),
        'avg_predicted_magnitude_%': round(avg_pred_mag, 3)
    }

    print(f"OK  ${start_price:.4f} -> ${end_price:.4f} ({pct_change:+.2f}%)")
    return coin_results, summary


def compute_robust_ratio(wrapped_coin, parent_coin, df, expected_min, expected_max):
    """Compute price ratio using median across ALL timestamps, not just last price.
    Validates against expected range. Falls back to range midpoint if data is bad."""
    w_df = df[df['coin_id'] == wrapped_coin].sort_values('timestamp_utc')
    p_df = df[df['coin_id'] == parent_coin].sort_values('timestamp_utc')

    if w_df.empty or p_df.empty:
        return (expected_min + expected_max) / 2, "DEFAULT"

    # Round timestamps and merge to find overlapping hours
    w_df = w_df.copy()
    p_df = p_df.copy()
    w_df['hour'] = w_df['timestamp_utc'].dt.round('h')
    p_df['hour'] = p_df['timestamp_utc'].dt.round('h')

    merged = w_df[['hour', 'price_usd']].merge(
        p_df[['hour', 'price_usd']], on='hour', suffixes=('_w', '_p'))

    if merged.empty:
        return (expected_min + expected_max) / 2, "DEFAULT"

    # Filter out zero prices
    merged = merged[(merged['price_usd_w'] > 0) & (merged['price_usd_p'] > 0)]
    if merged.empty:
        return (expected_min + expected_max) / 2, "DEFAULT"

    ratios = merged['price_usd_w'] / merged['price_usd_p']
    median_ratio = float(ratios.median())

    # Validate against expected range
    if expected_min <= median_ratio <= expected_max:
        return median_ratio, "OK"
    else:
        # Data is bad, use the expected range midpoint
        fallback = (expected_min + expected_max) / 2
        return fallback, f"BAD({median_ratio:.3f}->fixed {fallback:.3f})"


def predict_wrapped_token(wrapped_coin, parent_coin, parent_results, df, info):
    """Predict wrapped token using robust ratio from historical data."""
    if parent_results is None:
        return None, None

    ratio, ratio_status = compute_robust_ratio(
        wrapped_coin, parent_coin, df,
        info['min_ratio'], info['max_ratio'])

    wrapped_results = []
    wrapped_prices = []
    for r in parent_results:
        w_price = r['price_usd'] * ratio
        wrapped_prices.append(w_price)
        wrapped_results.append({
            'timestamp_utc': r['timestamp_utc'],
            'coin_id': wrapped_coin,
            'price_usd': w_price,
            'error': r['error'] * ratio,
            'total_volume_usd': 0.0,
            'hourly_increase': r['hourly_increase'] * ratio,
            'price-soc correlation': 0.0,
            'price-pop': 0.0,
            'predicted_social_volume': 0.0,
            'predicted_magnitude_pct': r['predicted_magnitude_pct']
        })

    start_price = wrapped_prices[0]
    end_price = wrapped_prices[-1]
    pct_change = ((end_price - start_price) / start_price) * 100

    summary = {
        'coin_id': wrapped_coin,
        'start_prediction_usd': start_price,
        'end_prediction_usd': end_price,
        'predicted_change_%': round(pct_change, 2),
        'avg_predicted_social': 0.0,
        'avg_predicted_magnitude_%': 0.0
    }

    print(f"    -> {wrapped_coin:<30} LINKED to {parent_coin} "
          f"(ratio={ratio:.4f} [{ratio_status}]) "
          f"OK  ${start_price:.4f} -> ${end_price:.4f} ({pct_change:+.2f}%)")

    return wrapped_results, summary


# =========================================================================
#  MAIN
# =========================================================================

def run_prediction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(">>> [1] Incarcare date istorice...")
    df = load_all_historical_data()
    if df is None or df.empty:
        return

    coins = df['coin_id'].unique()
    all_results = []
    summary_results = []

    independent_coins = [c for c in coins if c not in WRAPPED_TOKENS]
    wrapped_coins = [c for c in coins if c in WRAPPED_TOKENS]

    print(f"\n>>> [2] STAGE 1: Social volume (holdout bias correction)")
    print(f">>> [3] STAGE 2: Magnitude prediction")
    print(f">>> [4] STAGE 3: Price prediction (Monte Carlo)")
    print(f">>>     {len(independent_coins)} independent + "
          f"{len(wrapped_coins)} wrapped = {len(coins)} total | "
          f"{HORIZON}h | {NUM_SIMULATIONS} sims\n")

    parent_results = {}

    for coin in independent_coins:
        c_df = df[df['coin_id'] == coin]
        results, summary = predict_coin(coin, c_df, df)
        if results:
            all_results.extend(results)
            summary_results.append(summary)
            if coin in [v['parent'] for v in WRAPPED_TOKENS.values()]:
                parent_results[coin] = results

    # Wrapped tokens with robust ratio
    print(f"\n    --- Wrapped/staked tokens (linked with robust ratio) ---")

    for w_coin in wrapped_coins:
        info = WRAPPED_TOKENS[w_coin]
        parent = info['parent']
        if parent in parent_results:
            results, summary = predict_wrapped_token(
                w_coin, parent, parent_results[parent], df, info)
            if results:
                all_results.extend(results)
                summary_results.append(summary)
        else:
            print(f"    -> {w_coin:<30} parent {parent} not available, predicting independently")
            c_df = df[df['coin_id'] == w_coin]
            results, summary = predict_coin(w_coin, c_df, df)
            if results:
                all_results.extend(results)
                summary_results.append(summary)

    if not all_results:
        return

    # -- Export --
    res_df = pd.DataFrame(all_results)
    res_df['d_str'] = res_df['timestamp_utc'].dt.strftime('%Y-%m-%d')

    print(f"\n>>> [5] Salvare fisiere zilnice...")
    final_cols = ['timestamp_utc', 'coin_id', 'price_usd', 'error', 'total_volume_usd',
                  'hourly_increase', 'price-soc correlation', 'price-pop',
                  'predicted_social_volume', 'predicted_magnitude_pct']

    for date_val, data in res_df.groupby('d_str'):
        save_path = os.path.join(OUTPUT_DIR, f"{date_val}.csv")
        data[final_cols].to_csv(save_path, index=False)
        print(f"    Salvat: {date_val}.csv")

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(OUTPUT_DIR, "prediction_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n>>> [6] Fisier sumar salvat: prediction_summary.csv")


if __name__ == "__main__":
    run_prediction()
