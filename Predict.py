"""
Predict.py v7
─────────────
3-stage crypto price prediction:
  Stage 1: Social Volume — Seasonal Median (day-of-week + EWM baseline)
  Stage 2: Magnitude — XGBoost (how much will it move?)
  Stage 3: Price — XGBoost + Ornstein-Uhlenbeck Monte Carlo

Anti-bias protections:
  - Zero drift (no directional bias from training period)
  - Mean-reverting noise (Ornstein-Uhlenbeck process, standard in quant finance)
  - Per-hour clamp (±3% max move per hour, prevents feedback loops)
  - Anchor clamp (±30% max deviation over 7 days)
"""

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
WHALE_DIR = os.path.join(os.getcwd(), 'whale_data')
WHALE_SCORE_FILE = os.path.join(WHALE_DIR, 'daily_scores.csv')

HORIZON = 168  # 7 days
NUM_SIMULATIONS = 50
MAX_HOURLY_MOVE = 0.03   # ±3% per hour max
MAX_TOTAL_MOVE = 0.30    # ±30% total max from anchor
MEAN_REVERSION = 0.003   # Ornstein-Uhlenbeck theta: 0.3% pull per hour

PRICE_FEATURES = ['total_volume_usd', 'social_volume', 'hourly_increase', 'price-pop',
                   'price-soc correlation', 'RSI', 'volatility']

MAGNITUDE_FEATURES = ['volatility', 'RSI', 'social_volume', 'total_volume_usd',
                       'abs_change_lag1', 'abs_change_lag24', 'hour', 'day_of_week',
                       'whale_pressure', 'whale_activity']

WRAPPED_TOKENS = {
    'staked-ether':         {'parent': 'ethereum',  'min_ratio': 0.95, 'max_ratio': 1.05},
    'wrapped-steth':        {'parent': 'ethereum',  'min_ratio': 0.80, 'max_ratio': 1.25},
    'weth':                 {'parent': 'ethereum',  'min_ratio': 0.99, 'max_ratio': 1.01},
    'coinbase-wrapped-btc': {'parent': 'bitcoin',   'min_ratio': 0.95, 'max_ratio': 1.05},
    'jito-staked-sol':      {'parent': 'solana',    'min_ratio': 1.00, 'max_ratio': 1.50},
    'wrapped-eeth':         {'parent': 'ethereum',  'min_ratio': 0.95, 'max_ratio': 1.15},
}


# ── Data Loading ─────────────────────────────────────────────────────────────

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

    # Merge whale data
    full_df['date_key'] = full_df['timestamp_utc'].dt.strftime('%Y-%m-%d')
    if os.path.exists(WHALE_SCORE_FILE):
        whale_df = pd.read_csv(WHALE_SCORE_FILE)
        whale_cols = [c for c in ['date', 'whale_pressure', 'whale_activity'] if c in whale_df.columns]
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


# ── Dynamic Indicators ───────────────────────────────────────────────────────

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
#  STAGE 1: Social Volume — Seasonal Median
#
#  Standard time series decomposition:
#    prediction = baseline_level × seasonal_factor
#
#  baseline = Exponentially Weighted Median (recent weeks count more)
#  seasonal_factor = day_of_week_median / overall_median
#
#  Why NOT XGBoost for social:
#  - XGBoost autoregressive model feeds predictions back as lag features
#  - Over 7 days (168 steps), small errors compound exponentially
#  - Bitcoin social overpredicted 5000 when real was 1000
#  - Seasonal median is immune: no feedback loop, anchored to real data
# =========================================================================

def predict_social_seasonal(c_df, horizon=HORIZON):
    """Predict social volume using Seasonal Median decomposition."""
    sdf = c_df.copy()
    sdf['date'] = sdf['timestamp_utc'].dt.date

    # Aggregate to daily
    daily = sdf.groupby('date').agg(
        social_volume=('social_volume', 'first')
    ).reset_index().sort_values('date')

    daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek

    # Filter out zeros for statistics
    non_zero = daily[daily['social_volume'] > 0]['social_volume']
    if len(non_zero) < 3:
        # Not enough social data — return zeros
        return [0.0] * horizon, "NONE"

    # Baseline: EWM of last 21 days (3 weeks, recent emphasis)
    recent = daily['social_volume'].tail(21)
    ewm_baseline = float(recent.ewm(span=7, min_periods=1).mean().iloc[-1])
    overall_median = float(non_zero.median())

    # Safety: if EWM drifted too far from median, pull it back
    if overall_median > 0:
        ratio = ewm_baseline / overall_median
        if ratio > 2.5:
            ewm_baseline = overall_median * 2.0
        elif ratio < 0.4:
            ewm_baseline = overall_median * 0.5

    # Day-of-week seasonal factors
    dow_data = daily[daily['social_volume'] > 0]
    dow_medians = dow_data.groupby('dow')['social_volume'].median()
    overall_dow_median = float(dow_medians.median()) if len(dow_medians) > 0 else 1.0

    # Generate predictions
    last_ts = c_df['timestamp_utc'].iloc[-1]
    num_days = (horizon + 23) // 24
    daily_preds = []

    for d in range(num_days):
        t_day = last_ts + timedelta(days=d + 1)
        dow = t_day.weekday()

        # Seasonal factor
        if dow in dow_medians.index and overall_dow_median > 0:
            seasonal_factor = float(dow_medians[dow]) / overall_dow_median
        else:
            seasonal_factor = 1.0

        # Clamp seasonal factor to reasonable range
        seasonal_factor = max(0.3, min(seasonal_factor, 3.0))

        pred = ewm_baseline * seasonal_factor
        daily_preds.append(max(pred, 0.0))

    # Expand daily -> hourly
    hourly_preds = []
    for dv in daily_preds:
        hourly_preds.extend([dv] * 24)

    avg = np.mean(daily_preds)
    tag = f"SM({avg:.0f})"
    return hourly_preds[:horizon], tag


# =========================================================================
#  STAGE 2: Magnitude Prediction (XGBoost — how big will the move be?)
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
    last_wp = float(c_df['whale_pressure'].iloc[-1]) if 'whale_pressure' in c_df.columns else 0.0
    last_wa = float(c_df['whale_activity'].iloc[-1]) if 'whale_activity' in c_df.columns else 0.0

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
            'whale_pressure': last_wp, 'whale_activity': last_wa
        }])

        pred_mag = max(float(mag_model.predict(inp)[0]), 0.0005)
        predicted.append(pred_mag)
        mag_hist.append(pred_mag)

    return predicted


# =========================================================================
#  STAGE 3: Price — XGBoost + Ornstein-Uhlenbeck Monte Carlo
#
#  Ornstein-Uhlenbeck process (standard mean-reverting model):
#    dP = theta * (mu - P) * dt + sigma * dW
#
#  theta = MEAN_REVERSION (reversion speed, 0.3% per hour)
#  mu    = anchor_price (last known real price)
#  sigma = magnitude-scaled noise
#
#  This prevents runaway drift while allowing real price discovery.
# =========================================================================

def predict_coin(coin, c_df, df_all):
    c_df = c_df.copy().sort_values('timestamp_utc')
    c_df['lag_1'] = c_df['price_usd'].shift(1)
    c_df['lag_24'] = c_df['price_usd'].shift(24)

    train_data = c_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if len(train_data) < 50:
        print(f"    ! Skip {coin} -- not enough data ({len(train_data)} rows)")
        return None, None

    # -- Stage 1: Social (Seasonal Median) --
    social_preds, soc_tag = predict_social_seasonal(c_df, HORIZON)

    # -- Stage 2: Magnitude --
    mag_model, avg_mag = train_magnitude_model(c_df)
    mag_preds = predict_magnitude_series(mag_model, c_df, social_preds, avg_mag, HORIZON)
    mag_tag = "XGB" if mag_model else "AVG"
    avg_pred_mag = np.mean(mag_preds) * 100
    avg_pred_soc = np.mean(social_preds)

    print(f"    -> {coin:<30} soc={soc_tag} "
          f"mag={mag_tag}({avg_pred_mag:.2f}%) train={len(train_data)}h  ", end="")

    # -- Stage 3: Price (XGBoost + OU Monte Carlo) --
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

    anchor_price = float(c_df['price_usd'].iloc[-1])
    base_hist = np.array(c_df['price_usd'].tail(50).tolist())
    price_mat = np.tile(base_hist, (NUM_SIMULATIONS, 1))

    s_vol = last_row['total_volume_usd']
    s_pop = last_row['price-pop']
    s_corr = last_row['price-soc correlation']

    social_median = max(float(c_df['social_volume'].median()), 1.0)
    last_whale_a = float(c_df['whale_activity'].iloc[-1]) if 'whale_activity' in c_df.columns else 0.0

    coin_results = []
    coin_predicted_prices = []

    for h in range(HORIZON):
        t_curr += timedelta(hours=1)
        s_soc = social_preds[h]
        expected_magnitude = mag_preds[h]

        # Social magnitude boost
        social_ratio = s_soc / social_median if social_median > 0 else 1.0
        if social_ratio > 1.5:
            social_mag_boost = 1.0 + 0.4 * np.log2(social_ratio)
            expected_magnitude *= min(social_mag_boost, 3.0)

        # Whale activity boost
        if last_whale_a > 0.3:
            whale_boost = 1.0 + last_whale_a * 0.5
            expected_magnitude *= min(whale_boost, 2.0)

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

        # ── Ornstein-Uhlenbeck Monte Carlo ──
        # Noise: zero-mean, magnitude-scaled
        capped_mag = min(expected_magnitude, 0.04)
        noise_scale = capped_mag * anchor_price
        noise = np.random.normal(0.0, max(noise_scale, 1e-9), size=NUM_SIMULATIONS)

        # Mean reversion pull: gently pull predictions toward anchor
        # This is the theta*(mu - P) term of Ornstein-Uhlenbeck
        reversion_pull = MEAN_REVERSION * (anchor_price - base_preds)

        raw_preds = base_preds + noise + reversion_pull

        # Per-hour clamp: no single hour can move more than ±3% from previous
        raw_preds = np.clip(raw_preds,
                            l1 * (1 - MAX_HOURLY_MOVE),
                            l1 * (1 + MAX_HOURLY_MOVE))

        # Anchor clamp: total move ±30% from anchor over entire 7 days
        raw_preds = np.clip(raw_preds,
                            anchor_price * (1 - MAX_TOTAL_MOVE),
                            anchor_price * (1 + MAX_TOTAL_MOVE))

        raw_preds = np.maximum(raw_preds, 1e-9)

        mean_price = np.mean(raw_preds)
        std_error = np.std(raw_preds)
        mean_inc = np.mean(raw_preds - l1)

        price_mat = np.column_stack((price_mat[:, 1:], raw_preds))
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


# ── Wrapped Token Linking ────────────────────────────────────────────────────

def compute_robust_ratio(wrapped_coin, parent_coin, df, expected_min, expected_max):
    w_df = df[df['coin_id'] == wrapped_coin].sort_values('timestamp_utc')
    p_df = df[df['coin_id'] == parent_coin].sort_values('timestamp_utc')

    if w_df.empty or p_df.empty:
        return (expected_min + expected_max) / 2, "DEFAULT"

    w_df = w_df.copy()
    p_df = p_df.copy()
    w_df['hour'] = w_df['timestamp_utc'].dt.round('h')
    p_df['hour'] = p_df['timestamp_utc'].dt.round('h')

    merged = w_df[['hour', 'price_usd']].merge(
        p_df[['hour', 'price_usd']], on='hour', suffixes=('_w', '_p'))
    merged = merged[(merged['price_usd_w'] > 0) & (merged['price_usd_p'] > 0)]

    if merged.empty:
        return (expected_min + expected_max) / 2, "DEFAULT"

    ratios = merged['price_usd_w'] / merged['price_usd_p']
    median_ratio = float(ratios.median())

    if expected_min <= median_ratio <= expected_max:
        return median_ratio, "OK"
    else:
        fallback = (expected_min + expected_max) / 2
        return fallback, f"BAD({median_ratio:.3f}->fixed {fallback:.3f})"


def predict_wrapped_token(wrapped_coin, parent_coin, parent_results, df, info):
    if parent_results is None:
        return None, None

    ratio, ratio_status = compute_robust_ratio(
        wrapped_coin, parent_coin, df, info['min_ratio'], info['max_ratio'])

    wrapped_results = []
    wrapped_prices = []
    for r in parent_results:
        w_price = r['price_usd'] * ratio
        wrapped_prices.append(w_price)
        wrapped_results.append({
            'timestamp_utc': r['timestamp_utc'], 'coin_id': wrapped_coin,
            'price_usd': w_price, 'error': r['error'] * ratio,
            'total_volume_usd': 0.0, 'hourly_increase': r['hourly_increase'] * ratio,
            'price-soc correlation': 0.0, 'price-pop': 0.0,
            'predicted_social_volume': 0.0,
            'predicted_magnitude_pct': r['predicted_magnitude_pct']
        })

    start_price = wrapped_prices[0]
    end_price = wrapped_prices[-1]
    pct_change = ((end_price - start_price) / start_price) * 100

    summary = {
        'coin_id': wrapped_coin,
        'start_prediction_usd': start_price, 'end_prediction_usd': end_price,
        'predicted_change_%': round(pct_change, 2),
        'avg_predicted_social': 0.0, 'avg_predicted_magnitude_%': 0.0
    }

    print(f"    -> {wrapped_coin:<30} LINKED to {parent_coin} "
          f"(ratio={ratio:.4f} [{ratio_status}]) "
          f"OK  ${start_price:.4f} -> ${end_price:.4f} ({pct_change:+.2f}%)")

    return wrapped_results, summary


# ── Main ─────────────────────────────────────────────────────────────────────

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

    print(f"\n>>> [2] STAGE 1: Social (Seasonal Median — no overprediction)")
    print(f">>> [3] STAGE 2: Magnitude (XGBoost + whale features)")
    print(f">>> [4] STAGE 3: Price (XGBoost + Ornstein-Uhlenbeck Monte Carlo)")
    print(f">>>     {len(independent_coins)} independent + "
          f"{len(wrapped_coins)} wrapped = {len(coins)} total | "
          f"{HORIZON}h | {NUM_SIMULATIONS} sims | "
          f"reversion={MEAN_REVERSION} hourly_clamp={MAX_HOURLY_MOVE}\n")

    parent_results = {}

    for coin in independent_coins:
        c_df = df[df['coin_id'] == coin]
        results, summary = predict_coin(coin, c_df, df)
        if results:
            all_results.extend(results)
            summary_results.append(summary)
            if coin in [v['parent'] for v in WRAPPED_TOKENS.values()]:
                parent_results[coin] = results

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
