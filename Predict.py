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

SOCIAL_FEATURES = ['social_lag_1', 'social_lag_24', 'price_pct', 'total_volume_usd',
                    'RSI', 'volatility', 'hour', 'day_of_week']

# Magnitude model: predicts abs(hourly_pct_change) — "how big will the move be?"
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


# ── dynamic indicators for Monte Carlo ──────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1: Social Volume Prediction
# ═══════════════════════════════════════════════════════════════════════════

def train_social_model(c_df):
    sdf = c_df.copy()
    sdf['social_lag_1'] = sdf['social_volume'].shift(1)
    sdf['social_lag_24'] = sdf['social_volume'].shift(24)
    sdf['price_pct'] = sdf['price_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    sdf['hour'] = sdf['timestamp_utc'].dt.hour
    sdf['day_of_week'] = sdf['timestamp_utc'].dt.dayofweek
    sdf = sdf.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if len(sdf) < 50:
        return None

    X = sdf[SOCIAL_FEATURES]
    y = sdf['social_volume']

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, n_jobs=-1)
    try:
        model.fit(X, y)
        return model
    except:
        return None


def predict_social_series(social_model, c_df, horizon=HORIZON):
    if social_model is None:
        last_soc = c_df['social_volume'].iloc[-1]
        return [last_soc] * horizon

    social_hist = list(c_df['social_volume'].tail(24).values)
    last_vol = c_df['total_volume_usd'].iloc[-1]
    last_rsi = c_df['RSI'].iloc[-1]
    last_volat = c_df['volatility'].iloc[-1]
    last_ts = c_df['timestamp_utc'].iloc[-1]

    predicted = []
    for h in range(horizon):
        t_curr = last_ts + timedelta(hours=h + 1)
        s_lag_1 = social_hist[-1]
        s_lag_24 = social_hist[-24] if len(social_hist) >= 24 else social_hist[0]

        inp = pd.DataFrame([{
            'social_lag_1': s_lag_1,
            'social_lag_24': s_lag_24,
            'price_pct': 0.0,
            'total_volume_usd': last_vol,
            'RSI': last_rsi,
            'volatility': last_volat,
            'hour': t_curr.hour,
            'day_of_week': t_curr.dayofweek
        }])

        pred_soc = max(float(social_model.predict(inp)[0]), 0.0)
        predicted.append(pred_soc)
        social_hist.append(pred_soc)

    return predicted


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2: Magnitude Prediction — "how big will the move be?"
# ═══════════════════════════════════════════════════════════════════════════

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

    # Recent magnitude history
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

        pred_mag = max(float(mag_model.predict(inp)[0]), 0.0005)  # min 0.05%
        predicted.append(pred_mag)
        mag_hist.append(pred_mag)

    return predicted


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3: Price Prediction with magnitude-scaled Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

def run_prediction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(">>> [1] Incarcare date istorice...")
    df = load_all_historical_data()
    if df is None or df.empty:
        return

    coins = df['coin_id'].unique()
    all_results = []
    summary_results = []

    print(f"\n>>> [2] STAGE 1: Social volume prediction")
    print(f">>> [3] STAGE 2: Magnitude prediction (abs move size)")
    print(f">>> [4] STAGE 3: Price prediction (Monte Carlo)")
    print(f">>>     {len(coins)} monede | {HORIZON}h orizont | {NUM_SIMULATIONS} simulari\n")

    for coin in coins:
        c_df = df[df['coin_id'] == coin].copy().sort_values('timestamp_utc')
        c_df['lag_1'] = c_df['price_usd'].shift(1)
        c_df['lag_24'] = c_df['price_usd'].shift(24)

        train_data = c_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if len(train_data) < 50:
            print(f"    ! Skip {coin} — not enough data ({len(train_data)} rows)")
            continue

        # ── Stage 1: Social ──────────────────────────────────────────
        social_model = train_social_model(c_df)
        social_preds = predict_social_series(social_model, c_df, HORIZON)
        soc_tag = "XGB" if social_model else "FLAT"

        # ── Stage 2: Magnitude ───────────────────────────────────────
        mag_model, avg_mag = train_magnitude_model(c_df)
        mag_preds = predict_magnitude_series(mag_model, c_df, social_preds, avg_mag, HORIZON)
        mag_tag = "XGB" if mag_model else "AVG"
        avg_pred_mag = np.mean(mag_preds) * 100  # as percentage

        print(f"    -> {coin:<30} soc={soc_tag} mag={mag_tag}({avg_pred_mag:.2f}%) "
              f"train={len(train_data)}h  ", end="")

        # ── Stage 3: Price ───────────────────────────────────────────
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
            expected_magnitude = mag_preds[h]  # abs(pct_change) expected

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

            # MAGNITUDE-SCALED noise with safety caps:
            # 1. Cap magnitude at 2% per hour (no coin moves 50% in 1 hour normally)
            # 2. Use ANCHOR price for noise scale, not current (prevents compounding)
            # 3. Clamp final price to +-50% of starting price
            drift_array = np.full(NUM_SIMULATIONS, historical_drift)
            capped_mag = min(expected_magnitude, 0.02)  # max 2% per hour
            anchor_price = base_hist[-1]  # starting price, constant
            noise_scale = capped_mag * anchor_price  # fixed scale, no compounding
            noise = np.random.normal(drift_array, max(noise_scale, 1e-9))
            final_preds = np.maximum(base_preds + noise, 1e-9)
            # Clamp to +-50% of starting price to prevent runaway
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
            'avg_predicted_social': round(np.mean(social_preds), 2),
            'avg_predicted_magnitude_%': round(avg_pred_mag, 3)
        })

        print(f"OK  ${start_price:.4f} -> ${end_price:.4f} ({pct_change:+.2f}%)")

    if not all_results:
        return

    # ── Export ────────────────────────────────────────────────────────
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
