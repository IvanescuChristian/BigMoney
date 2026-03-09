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

HORIZON = 168 # 7 Zile in ore
NUM_SIMULATIONS = 50 # Rute Monte Carlo
FEATURES = ['total_volume_usd', 'social_volume', 'hourly_increase', 'price-pop', 
            'price-soc correlation', 'RSI', 'volatility']

def load_all_historical_data():
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not all_files: return None
    
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            
            # Găsim coloana de social, dar nu depindem de ea
            sv_col = [c for c in temp_df.columns if 'social_volume' in c]
            if sv_col: temp_df['social_volume'] = temp_df[sv_col[0]]
            else: temp_df['social_volume'] = 0.0
            
            needed = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd', 
                      'social_volume', 'hourly_increase', 'price-soc correlation', 
                      'price-pop', 'RSI', 'volatility']
            
            for col in needed:
                if col not in temp_df.columns: temp_df[col] = 0.0
                    
            df_list.append(temp_df[needed])
        except: pass
            
    if not df_list: return None
        
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp_utc'] = pd.to_datetime(full_df['timestamp_utc'])
    full_df = full_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return full_df.sort_values(['coin_id', 'timestamp_utc'])

def get_rsi_2d(prices_matrix, period=14):
    if prices_matrix.shape[1] < period + 1: return np.full(prices_matrix.shape[0], 50.0)
    deltas = np.diff(prices_matrix[:, -period-1:], axis=1)
    ups = np.sum(np.where(deltas > 0, deltas, 0), axis=1) / period
    downs = np.sum(np.where(deltas < 0, -deltas, 0), axis=1) / period
    downs = np.where(downs == 0, 1e-9, downs)
    rsi = 100.0 - (100.0 / (1.0 + (ups / downs)))
    return np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)

def get_volat_2d(prices_matrix, period=24):
    if prices_matrix.shape[1] < period: return np.zeros(prices_matrix.shape[0])
    arr = prices_matrix[:, -period:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(arr, axis=1) / (arr[:, :-1] + 1e-9)
    return np.nan_to_num(np.std(rets, axis=1), nan=0.0, posinf=0.0, neginf=0.0)

def run_prediction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(">>> [1] Incarcare date istorice...")
    df = load_all_historical_data()
    if df is None or df.empty: return

    coins = df['coin_id'].unique()
    all_results = []
    print(f">>> [2] Predictie Monte Carlo pentru {len(coins)} monede...")

    for coin in coins:
        c_df = df[df['coin_id'] == coin].copy().sort_values('timestamp_utc')
        c_df['lag_1'] = c_df['price_usd'].shift(1)
        c_df['lag_24'] = c_df['price_usd'].shift(24)
        
        train_data = c_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if len(train_data) < 50:
            print(f"    ! Lipsesc date pentru {coin}. Sarim.")
            continue

        X_cols = FEATURES + ['lag_1', 'lag_24']
        X = train_data[X_cols]
        y = train_data['price_usd']
        
        model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=6, n_jobs=-1)
        
        try:
            model.fit(X, y)
        except Exception as e:
            print(f"    ! Eroare XGBoost ({coin}): {e}")
            continue

        last_row = c_df.iloc[-1]
        t_curr = last_row['timestamp_utc']
        
        base_hist = np.array(c_df['price_usd'].tail(50).tolist())
        price_mat = np.tile(base_hist, (NUM_SIMULATIONS, 1))
        
        s_vol = last_row['total_volume_usd']
        s_soc = last_row['social_volume']
        s_pop = last_row['price-pop']
        s_corr = last_row['price-soc correlation']

        for _ in range(HORIZON):
            t_curr += timedelta(hours=1)
            
            l1 = price_mat[:, -1]
            l24 = price_mat[:, -24] if price_mat.shape[1] >= 24 else price_mat[:, 0]
            inc = l1 - price_mat[:, -2]
            
            rsi_vec = get_rsi_2d(price_mat)
            volat_vec = get_volat_2d(price_mat)
            
            # --- FIX: BASELINE VOLATILITY ---
            # Daca datele nu arata volatilitate (lipsa date/stagnare API),
            # impunem un minim de miscare de 0.15% pentru a nu bloca Monte Carlo.
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
            
            base_preds = model.predict(inp)
            
            # Zgomot calculat pe baza volatilitatii (reale sau impuse)
            noise = np.random.normal(0, volat_vec * base_preds * 0.05) 
            final_preds = np.maximum(base_preds + noise, 1e-9)
            
            mean_price = np.mean(final_preds)
            std_error = np.std(final_preds)
            mean_inc = np.mean(final_preds - l1)
            
            price_mat = np.column_stack((price_mat[:, 1:], final_preds))
            
            all_results.append({
                'timestamp_utc': t_curr,
                'coin_id': coin,
                'price_usd': mean_price,
                'error': std_error,
                'total_volume_usd': s_vol,
                'hourly_increase': mean_inc,
                'price-soc correlation': s_corr,
                'price-pop': s_pop,
                'social_volume_raw': s_soc
            })
        print(f"    [OK] {coin}")

    if not all_results: return
    
    res_df = pd.DataFrame(all_results)
    res_df['d_str'] = res_df['timestamp_utc'].dt.strftime('%Y-%m-%d')
    
    print("\n>>> [3] Salvare in predicted_hourly...")
    final_cols = ['timestamp_utc', 'coin_id', 'price_usd', 'error', 'total_volume_usd', 
                  'hourly_increase', 'price-soc correlation', 'price-pop']
                  
    for date_val, data in res_df.groupby('d_str'):
        sv_name = f"social_volume_{date_val.replace('-', '_')}"
        data[sv_name] = data['social_volume_raw']
        cols_to_save = final_cols + [sv_name]
        
        save_path = os.path.join(OUTPUT_DIR, f"{date_val}.csv")
        data[cols_to_save].to_csv(save_path, index=False)
        print(f"    Salvat: {date_val}.csv")
        
if __name__ == "__main__":
    run_prediction()