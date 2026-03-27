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

HORIZON = 168  # 7 zile in ore
NUM_SIMULATIONS = 50

PRICE_FEATURES = ['total_volume_usd', 'social_volume', 'hourly_increase', 'price-pop',
                   'price-soc correlation', 'RSI', 'volatility']

SOCIAL_FEATURES = ['social_lag_1d', 'social_lag_7d', 'social_avg_7d',
                    'price_pct_1d', 'volume_avg', 'day_of_week']

MAGNITUDE_FEATURES = ['volatility', 'RSI', 'social_volume', 'total_volume_usd',
                       'abs_change_lag1', 'abs_change_lag24', 'hour', 'day_of_week']


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
#  STAGE 1: Social Volume Prediction (DAILY, not hourly)
# =========================================================================

def train_social_model(c_df):
    """Train on DAILY social volume. Social is a daily metric, not hourly."""
    sdf = c_df.copy()
    sdf['date'] = sdf['timestamp_utc'].dt.date

    # Aggregate to daily
    daily = sdf.groupby('date').agg(
        social_volume=('social_volume', 'first'),
        price_start=('price_usd', 'first'),
        price_end=('price_usd', 'last'),
        volume_avg=('total_volume_usd', 'mean')
    ).reset_index().sort_values('date')

    # Daily features
    daily['social_lag_1d'] = daily['social_volume'].shift(1)
    daily['social_lag_7d'] = daily['social_volume'].shift(7)
    daily['social_avg_7d'] = daily['social_volume'].rolling(7, min_periods=1).mean()
    daily['price_pct_1d'] = daily['price_end'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    daily['day_of_week'] = pd.to_datetime(daily['date']).dt.dayofweek
    daily = daily.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if len(daily) < 10:
        return None, daily

    X = daily[SOCIAL_FEATURES]
    y = daily['social_volume']

    model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        reg_alpha=1.0, reg_lambda=2.0,
        n_jobs=-1
    )
    try:
        model.fit(X, y)
        return model, daily
    except:
        return None, daily


def predict_social_series(social_result, c_df, horizon=HORIZON):
    """Predict daily social volume for 7 days, expand to hourly."""
    # Handle None or invalid result
    if social_result is None:
        last_soc = max(float(c_df['social_volume'].iloc[-1]), 0.0)
        return [last_soc] * horizon

    social_model, daily = social_result

    if social_model is None:
        median_soc = max(float(daily['social_volume'].tail(7).median()), 0.0)
        return [median_soc] * horizon

    # Historical stats for clamping
    hist_median = float(daily['social_volume'].median())
    hist_max = float(daily['social_volume'].max())
    clamp_max = max(hist_max * 1.5, hist_median * 3, 1.0)

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

        pred_soc = max(float(social_model.predict(inp)[0]), 0.0)
        pred_soc = min(pred_soc, clamp_max)
        daily_preds.append(pred_soc)
        social_hist.append(pred_soc)

    # Expand daily -> hourly
    hourly_preds = []
    for dv in daily_preds:
        hourly_preds.extend([dv] * 24)
    return hourly_preds[:horizon]


# =========================================================================
#  STAGE 2: Magnitude Prediction
# =========================================================================

def train_magnitude_model(c_df):
    """Train model to predict abs(hourly_pct_change)."""
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
        avg_magnitude = float(y.mean())
        return model, avg_magnitude
    except:
        return None, float(y.mean()) if len(y) > 0 else 0.001


def predict_magnitude_series(mag_model, c_df, social_preds, avg_mag, horizon=HORIZON):
    """Predict expected abs(pct_change) for each future hour."""
    if mag_model is None:
        return [avg_mag] * horizon

    last_vol = c_df['total_volume_usd'].iloc[-1]
    last_rsi = c_df['RSI'].iloc[-1]
    last_volat = c_df['volatility'].iloc[-1]
    last_ts = c_df['timestamp_utc'].iloc[-1]

    pct_changes = c_df['price_usd'].pct_change().abs().fillna(0)
    mag_hist = list(pct_changes.tail(24).values)

    predicted = []
    for h in range(horizon):
        t_curr = last_ts + timedelta(hours=h + 1)
        lag1 = mag_hist[-1]
        lag24 = mag_hist[-24] if len(mag_hist) >= 24 else mag_hist[0]

        inp = pd.DataFrame([{
            'volatility': last_volat,
            'RSI': last_rsi,
            'social_volume': social_preds[h],
            'total_volume_usd': last_vol,
            'abs_change_lag1': lag1,
            'abs_change_lag24': lag24,
            'hour': t_curr.hour,
            'day_of_week': t_curr.dayofweek
        }])

        pred_mag = max(float(mag_model.predict(inp)[0]), 0.0005)
        predicted.append(pred_mag)
        mag_hist.append(pred_mag)

    return predicted


# =========================================================================
#  STAGE 3: Price Prediction with Monte Carlo
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

    print(f"\n>>> [2] STAGE 1: Social volume prediction (daily model)")
    print(f">>> [3] STAGE 2: Magnitude prediction (abs move size)")
    print(f">>> [4] STAGE 3: Price prediction (Monte Carlo)")
    print(f">>>     {len(coins)} monede | {HORIZON}h orizont | {NUM_SIMULATIONS} simulari\n")

    for coin in coins:
        c_df = df[df['coin_id'] == coin].copy().sort_values('timestamp_utc')
        c_df['lag_1'] = c_df['price_usd'].shift(1)
        c_df['lag_24'] = c_df['price_usd'].shift(24)

        train_data = c_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if len(train_data) < 50:
            print(f"    ! Skip {coin} -- not enough data ({len(train_data)} rows)")
            continue

        # -- Stage 1: Social (DAILY) --
        social_result = train_social_model(c_df)
        social_model_obj = social_result[0] if social_result[0] is not None else None
        social_preds = predict_social_series(social_result, c_df, HORIZON)
        soc_tag = "XGB" if social_model_obj is not None else "MED"

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
            continue

        last_row = c_df.iloc[-1]
        t_curr = last_row['timestamp_utc']
        historical_drift = train_data['hourly_increase'].mean()

        base_hist = np.array(c_df['price_usd'].tail(50).tolist())
        price_mat = np.tile(base_hist, (NUM_SIMULATIONS, 1))

        s_vol = last_row['total_volume_usd']
        s_pop = last_row['price-pop']
        s_corr = last_row['price-soc correlation']

        coin_predicted_prices = []

        for h in range(HORIZON):
            t_curr += timedelta(hours=1)

            s_soc = social_preds[h]
            expected_magnitude = mag_preds[h]

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
                'RSI': rsi_vec,
                'volatility': volat_vec,
                'lag_1': l1,
                'lag_24': l24
            })[X_cols]

            base_preds = price_model.predict(inp)

            # MAGNITUDE-SCALED noise with safety caps
            drift_array = np.full(NUM_SIMULATIONS, historical_drift)
            capped_mag = min(expected_magnitude, 0.02)
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

            all_results.append({
                'timestamp_utc': t_curr,
                'coin_id': coin,
                'price_usd': mean_price,
                'error': std_error,
                'total_volume_usd': s_vol,
                'hourly_increase': mean_inc,
                'price-soc correlation': s_corr,
                'price-pop': s_pop,
                'predicted_social_volume': s_soc,
                'predicted_magnitude_pct': expected_magnitude * 100
            })

        start_price = coin_predicted_prices[0]
        end_price = coin_predicted_prices[-1]
        pct_change = ((end_price - start_price) / start_price) * 100

        summary_results.append({
            'coin_id': coin,
            'start_prediction_usd': start_price,
            'end_prediction_usd': end_price,
            'predicted_change_%': round(pct_change, 2),
            'avg_predicted_social': round(avg_pred_soc, 2),
            'avg_predicted_magnitude_%': round(avg_pred_mag, 3)
        })

        print(f"OK  ${start_price:.4f} -> ${end_price:.4f} ({pct_change:+.2f}%)")

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
