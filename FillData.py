import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, 'historical_hourly')

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def process_data():
    print(f">>> [1] Incarcare date brute din fisiere...")
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not all_files:
        print("Eroare: Nu exista fisiere CSV in historical_hourly.")
        return

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            if df.empty: continue
            
            # Preluam coloana social_volume (indiferent de data din numele ei)
            social_cols = [c for c in df.columns if 'social_volume' in c]
            df['social_volume'] = df[social_cols[0]] if social_cols else 0.0
            
            expected_cols = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd', 'social_volume']
            for c in expected_cols:
                if c not in df.columns: df[c] = 0.0
            
            df_list.append(df[expected_cols])
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp_utc'] = pd.to_datetime(full_df['timestamp_utc'])
    
    # CHEIA: Rotunjim la cea mai apropiata ora ('h' mic pentru compatibilitate)
    full_df['timestamp_utc'] = full_df['timestamp_utc'].dt.round('h')
    full_df = full_df.drop_duplicates(subset=['coin_id', 'timestamp_utc'], keep='last')
    
    print(">>> [2] Construire Grila Master (Toate monedele / Toate orele)...")
    min_time = full_df['timestamp_utc'].min()
    max_time = full_df['timestamp_utc'].max()
    all_hours = pd.date_range(min_time, max_time, freq='h')
    all_coins = full_df['coin_id'].dropna().unique()
    
    # Cream grila si lipim datele
    grid = pd.MultiIndex.from_product([all_coins, all_hours], names=['coin_id', 'timestamp_utc']).to_frame(index=False)
    merged = pd.merge(grid, full_df, on=['coin_id', 'timestamp_utc'], how='left')
    
    print(">>> [3] Umplere date si generare indicatori (RSI, Volatilitate)...")

    def process_coin_group(group):
        group = group.sort_values('timestamp_utc').copy()
        
        # Umplem golurile cu datele cele mai apropiate
        cols_numeric = ['price_usd', 'total_volume_usd', 'social_volume']
        group[cols_numeric] = group[cols_numeric].interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0)
        
        # Indicatori
        group['hourly_increase'] = group['price_usd'].diff().fillna(0)
        group['RSI'] = calculate_rsi(group['price_usd'])
        
        p_pct = group['price_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        v_pct = group['total_volume_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        group['price-pop'] = p_pct.rolling(24, min_periods=1).corr(v_pct).fillna(0)
        group['price-soc correlation'] = group['price_usd'].rolling(24, min_periods=1).corr(group['social_volume']).fillna(0)
        group['volatility'] = p_pct.rolling(24, min_periods=1).std().fillna(0)
        
        return group

    # Iteram manual pentru a evita erorile de index din Pandas 2.2+
    processed_chunks = []
    for coin, group in merged.groupby('coin_id'):
        processed_chunks.append(process_coin_group(group))
        
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(">>> [4] Salvare fisiere in historical_hourly...")
    final_df['date_str'] = final_df['timestamp_utc'].dt.strftime('%Y-%m-%d')
    
    for d_str, day_data in final_df.groupby('date_str'):
        sv_name = f"social_volume_{d_str.replace('-', '_')}"
        day_data[sv_name] = day_data['social_volume']
        
        # Coloanele care vor fi suprascrise in historical (inclusiv RSI pentru Predict.py)
        cols_save = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd', 
                     'hourly_increase', 'price-soc correlation', 'price-pop', sv_name, 
                     'RSI', 'volatility']
        
        path = os.path.join(INPUT_DIR, f"{d_str}.csv")
        day_data[cols_save].to_csv(path, index=False)
        
    print(f">>> [SUCCESS] Procesat {len(all_coins)} monede pe {len(all_hours)} ore.")

if __name__ == "__main__":
    process_data()