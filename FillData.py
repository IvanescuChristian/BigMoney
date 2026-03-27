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


def fill_social_daily(coin_df, all_dates):
    """
    Fill social volume as a DAILY metric (not hourly).
    1. Extract one value per day from real data
    2. Interpolate missing days smoothly (cubic or linear)
    3. Verify no spikes (clamp jumps > 3x median)
    4. Expand back to hourly (same value all 24h)
    """
    coin_df = coin_df.sort_values('timestamp_utc').copy()
    coin_df['date'] = coin_df['timestamp_utc'].dt.date

    # Step 1: Extract daily social values (take first non-zero per day)
    daily_social = {}
    for date, group in coin_df.groupby('date'):
        vals = group['social_volume'].dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            daily_social[date] = float(vals.iloc[0])

    if not daily_social:
        # No social data at all for this coin — leave as 0
        coin_df['social_volume'] = 0.0
        return coin_df

    # Step 2: Build daily series and interpolate gaps
    date_range = sorted(all_dates)
    daily_series = pd.Series(index=date_range, dtype=float)
    for d, v in daily_social.items():
        if d in daily_series.index:
            daily_series[d] = v

    # Count real data points
    real_count = daily_series.notna().sum()
    total_days = len(daily_series)

    if real_count == total_days:
        # All days have data — no filling needed
        pass
    elif real_count >= 3:
        # Enough points for smooth interpolation
        daily_series = daily_series.interpolate(method='linear', limit_direction='both')
    elif real_count >= 1:
        # Very few points — forward/backward fill
        daily_series = daily_series.ffill().bfill()
    else:
        daily_series = daily_series.fillna(0.0)

    # Ensure no negatives
    daily_series = daily_series.clip(lower=0)

    # Step 3: Spike detection — no day should be > 3x the rolling median
    if real_count >= 5:
        rolling_med = daily_series.rolling(7, min_periods=1, center=True).median()
        spike_threshold = rolling_med * 3 + 1  # +1 for coins with very low volume
        # Only clamp interpolated values, not real data
        for d in date_range:
            if d not in daily_social and daily_series[d] > spike_threshold[d]:
                daily_series[d] = spike_threshold[d]

    # Step 4: Expand daily -> hourly (same value for all hours in a day)
    date_to_social = daily_series.to_dict()
    coin_df['social_volume'] = coin_df['date'].map(date_to_social).fillna(0.0)

    coin_df = coin_df.drop(columns=['date'])
    return coin_df


def process_data():
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

    if not all_files:
        print("Eroare: Nu exista fisiere CSV in historical_hourly.")
        return

    # Detect all dates from filenames
    all_date_strs = []
    for f in all_files:
        bn = os.path.splitext(os.path.basename(f))[0]
        all_date_strs.append(bn)

    print(f"Found {len(all_files)} files covering {all_date_strs[0]} to {all_date_strs[-1]}")

    # Load all data
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            if df.empty:
                continue

            social_cols = [c for c in df.columns if 'social_volume' in c]
            df['social_volume'] = df[social_cols[0]] if social_cols else 0.0

            expected_cols = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd', 'social_volume']
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = 0.0

            df_list.append(df[expected_cols])
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp_utc'] = pd.to_datetime(full_df['timestamp_utc'])
    full_df['timestamp_utc'] = full_df['timestamp_utc'].dt.round('h')
    full_df = full_df.drop_duplicates(subset=['coin_id', 'timestamp_utc'], keep='last')

    # Build master grid
    min_time = full_df['timestamp_utc'].min()
    max_time = full_df['timestamp_utc'].max()
    all_hours = pd.date_range(min_time, max_time, freq='h')
    all_coins = sorted(full_df['coin_id'].dropna().unique())
    all_dates = sorted(set(h.date() for h in all_hours))

    print(f"Found {len(all_coins)} distinct coins across {len(all_dates)} days.")

    grid = pd.MultiIndex.from_product([all_coins, all_hours],
                                       names=['coin_id', 'timestamp_utc']).to_frame(index=False)
    merged = pd.merge(grid, full_df, on=['coin_id', 'timestamp_utc'], how='left')

    # Process each coin
    processed_chunks = []
    social_stats = {'filled': 0, 'real': 0, 'zero': 0}

    for coin in all_coins:
        group = merged[merged['coin_id'] == coin].copy().sort_values('timestamp_utc')

        # --- PRICE: linear interpolation (existing logic, works fine) ---
        price_cols = ['price_usd', 'total_volume_usd']
        group[price_cols] = group[price_cols].interpolate(method='linear',
                                                           limit_direction='both').ffill().bfill().fillna(0)

        # --- SOCIAL: daily-aware filling (NEW) ---
        real_sv_count = (group['social_volume'].dropna() > 0).sum()
        group = fill_social_daily(group, all_dates)
        filled_sv_count = (group['social_volume'] > 0).sum()

        if real_sv_count > 0:
            social_stats['real'] += 1
            if filled_sv_count > real_sv_count:
                social_stats['filled'] += 1
        else:
            social_stats['zero'] += 1

        # --- INDICATORS ---
        group['hourly_increase'] = group['price_usd'].diff().fillna(0)
        group['RSI'] = calculate_rsi(group['price_usd'])

        p_pct = group['price_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        v_pct = group['total_volume_usd'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

        group['price-pop'] = p_pct.rolling(24, min_periods=1).corr(v_pct).fillna(0)
        group['price-soc correlation'] = group['price_usd'].rolling(24, min_periods=1).corr(
            group['social_volume']).fillna(0)
        group['volatility'] = p_pct.rolling(24, min_periods=1).std().fillna(0)

        processed_chunks.append(group)

    final_df = pd.concat(processed_chunks, ignore_index=True)

    print(f"\nSocial volume stats:")
    print(f"  Coins with real data: {social_stats['real']}")
    print(f"  Coins with gaps filled: {social_stats['filled']}")
    print(f"  Coins with no social data: {social_stats['zero']}")

    # Save per-day files
    print(f"\nSaving files...")
    final_df['date_str'] = final_df['timestamp_utc'].dt.strftime('%Y-%m-%d')

    saved = 0
    for d_str, day_data in final_df.groupby('date_str'):
        sv_name = f"social_volume_{d_str.replace('-', '_')}"
        day_data = day_data.copy()
        day_data[sv_name] = day_data['social_volume']

        cols_save = ['timestamp_utc', 'coin_id', 'price_usd', 'total_volume_usd',
                     'hourly_increase', 'price-soc correlation', 'price-pop', sv_name,
                     'RSI', 'volatility']

        path = os.path.join(INPUT_DIR, f"{d_str}.csv")
        day_data[cols_save].to_csv(path, index=False)
        coin_count = day_data['coin_id'].nunique()
        saved += 1
        print(f"  Saved {d_str} ({coin_count} coins)")

    print(f"\nDone.")


if __name__ == "__main__":
    process_data()
