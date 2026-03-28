"""
whale_collector.py
──────────────────
Collects Bitcoin on-chain metrics from blockchain.info Charts API.
COMPLETELY FREE — no API key, no rate limit, no proxy needed.

These aggregate daily metrics serve as whale activity signals:
  - estimated_btc_volume: total BTC moved on-chain (high = whales active)
  - n_transactions: transaction count (high = network busy)
  - avg_block_size: block fullness indicator
  - mempool_size: congestion (high = urgency)
  - hash_rate: miner confidence
  - estimated_usd_volume: total USD moved on-chain

Usage:
    python whale_collector.py              # auto-detect date range from historical_hourly/
    python whale_collector.py --days 60    # last 60 days
    python whale_collector.py --from 2025-03-31 --to 2025-05-24
"""

import requests
import pandas as pd
import numpy as np
import os
import sys
import time
import glob
from datetime import datetime, timedelta, timezone

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WHALE_DIR  = os.path.join(BASE_DIR, "whale_data")
SCORE_FILE = os.path.join(WHALE_DIR, "daily_scores.csv")

BLOCKCHAIN_API = "https://api.blockchain.info/charts"

# Charts to fetch — each becomes a daily feature for Predict.py
CHARTS = {
    'estimated-transaction-volume':     'est_btc_volume',
    'estimated-transaction-volume-usd': 'est_usd_volume',
    'n-transactions':                   'n_transactions',
    'avg-block-size':                   'avg_block_size',
    'mempool-size':                     'mempool_bytes',
    'hash-rate':                        'hash_rate',
    'n-transactions-per-block':         'tx_per_block',
}


def get_training_dates():
    """Auto-detect date range from historical_hourly/."""
    hist_dir = os.path.join(BASE_DIR, 'historical_hourly')
    if not os.path.exists(hist_dir):
        return None, None
    files = sorted(glob.glob(os.path.join(hist_dir, '*.csv')))
    dates = []
    for f in files:
        bn = os.path.splitext(os.path.basename(f))[0]
        try:
            datetime.strptime(bn, '%Y-%m-%d')
            dates.append(bn)
        except ValueError:
            pass
    if dates:
        return dates[0], dates[-1]
    return None, None


def fetch_chart(chart_name, start_date, end_date, timespan_days):
    """
    Fetch one chart from blockchain.info.
    Returns dict: {date_str: value}
    """
    url = f"{BLOCKCHAIN_API}/{chart_name}"
    params = {
        'timespan': f'{timespan_days}days',
        'start': start_date,
        'format': 'json',
        'sampled': 'false'  # get daily, not sampled
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f" HTTP {r.status_code}", end="")
            return {}

        data = r.json()
        values = data.get('values', [])

        result = {}
        for point in values:
            ts = point.get('x', 0)
            val = point.get('y', 0)
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
            result[date_str] = val

        return result

    except Exception as e:
        print(f" ERROR({e})", end="")
        return {}


def compute_whale_scores(daily_df):
    """
    From raw on-chain metrics, compute derived whale signals:
    - whale_pressure: volume spike relative to rolling average (high = whales active)
    - network_stress: mempool + tx count anomaly
    - miner_confidence: hashrate trend
    """
    if daily_df.empty:
        return daily_df

    df = daily_df.copy().sort_values('date')

    # Whale pressure: how much above average is today's volume?
    if 'est_btc_volume' in df.columns:
        rolling_avg = df['est_btc_volume'].rolling(14, min_periods=1).mean()
        df['whale_pressure'] = ((df['est_btc_volume'] - rolling_avg) / rolling_avg.clip(lower=1)).round(4)
    else:
        df['whale_pressure'] = 0.0

    # Network stress: above-average transactions + mempool
    if 'n_transactions' in df.columns:
        tx_avg = df['n_transactions'].rolling(14, min_periods=1).mean()
        df['tx_anomaly'] = ((df['n_transactions'] - tx_avg) / tx_avg.clip(lower=1)).round(4)
    else:
        df['tx_anomaly'] = 0.0

    if 'mempool_bytes' in df.columns:
        mem_avg = df['mempool_bytes'].rolling(14, min_periods=1).mean()
        df['mempool_anomaly'] = ((df['mempool_bytes'] - mem_avg) / mem_avg.clip(lower=1)).round(4)
    else:
        df['mempool_anomaly'] = 0.0

    # Miner confidence: hashrate trend (positive = growing network)
    if 'hash_rate' in df.columns:
        hr_avg = df['hash_rate'].rolling(14, min_periods=1).mean()
        df['miner_confidence'] = ((df['hash_rate'] - hr_avg) / hr_avg.clip(lower=1)).round(4)
    else:
        df['miner_confidence'] = 0.0

    # Combined whale activity score
    df['whale_activity'] = (
        df['whale_pressure'].abs() * 0.4 +
        df['tx_anomaly'].abs() * 0.3 +
        df['mempool_anomaly'].abs() * 0.2 +
        df['miner_confidence'].abs() * 0.1
    ).round(4)

    return df


def main():
    print("=" * 60)
    print("  WHALE TRACKER -- Bitcoin On-Chain Metrics")
    print("  blockchain.info Charts API (FREE, no key needed)")
    print("=" * 60)

    # Determine date range
    if '--from' in sys.argv and '--to' in sys.argv:
        from_idx = sys.argv.index('--from')
        to_idx = sys.argv.index('--to')
        start_date = sys.argv[from_idx + 1]
        end_date = sys.argv[to_idx + 1]
    elif '--days' in sys.argv:
        idx = sys.argv.index('--days')
        days = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 60
        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
    else:
        # Auto-detect from historical_hourly/
        start_date, end_date = get_training_dates()
        if not start_date:
            start_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime('%Y-%m-%d')
            end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    timespan_days = (end_dt - start_dt).days + 14  # extra margin

    print(f"\n  Date range: {start_date} -> {end_date} ({(end_dt - start_dt).days} days)")
    print(f"  Charts to fetch: {len(CHARTS)}")
    print(f"  No API key needed, no proxy needed\n")

    # Fetch all charts
    all_data = {}
    for chart_name, col_name in CHARTS.items():
        print(f"  [{list(CHARTS.keys()).index(chart_name)+1}/{len(CHARTS)}] {chart_name}...", end="", flush=True)
        values = fetch_chart(chart_name, start_date, end_date, timespan_days)
        print(f" {len(values)} days")
        all_data[col_name] = values
        time.sleep(1)  # be polite

    # Build daily DataFrame
    all_dates = set()
    for col_data in all_data.values():
        all_dates.update(col_data.keys())

    # Filter to requested range
    all_dates = sorted(d for d in all_dates if start_date <= d <= end_date)

    if not all_dates:
        print("\n  [ERROR] No data retrieved")
        return

    rows = []
    for date in all_dates:
        row = {'date': date}
        for col_name, col_data in all_data.items():
            row[col_name] = col_data.get(date, np.nan)
        rows.append(row)

    daily_df = pd.DataFrame(rows)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Fill NaN with interpolation
    numeric_cols = [c for c in daily_df.columns if c != 'date']
    daily_df[numeric_cols] = daily_df[numeric_cols].interpolate(method='linear').ffill().bfill()

    # Compute derived whale signals
    daily_df = compute_whale_scores(daily_df)

    # Save
    os.makedirs(WHALE_DIR, exist_ok=True)
    daily_df.to_csv(SCORE_FILE, index=False)

    print(f"\n  Saved: {SCORE_FILE} ({len(daily_df)} days)")

    # Print summary
    print(f"\n  {'Date':<12} {'BTC Vol':>12} {'USD Vol':>14} {'TXs':>10} {'Whale P':>9} {'Activity':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*10} {'-'*9} {'-'*10}")

    show_df = daily_df.tail(14)  # show last 2 weeks
    for _, row in show_df.iterrows():
        btc_vol = row.get('est_btc_volume', 0)
        usd_vol = row.get('est_usd_volume', 0)
        txs = row.get('n_transactions', 0)
        wp = row.get('whale_pressure', 0)
        wa = row.get('whale_activity', 0)

        signal = "HIGH" if wa > 0.3 else ("MED" if wa > 0.1 else "LOW")

        print(f"  {row['date']:<12} {btc_vol:>11,.0f} ${usd_vol:>12,.0f} "
              f"{txs:>9,.0f} {wp:>+8.3f}  {wa:>8.3f} {signal}")

    # Stats
    print(f"\n  Metrics available per day:")
    for col in numeric_cols:
        if col in daily_df.columns:
            print(f"    {col:<25} avg={daily_df[col].mean():>12,.2f}")

    print(f"\n{'=' * 60}")
    print(f"  Ready to integrate into Predict.py as features!")
    print(f"  whale_pressure: volume spike (high = whales moving BTC)")
    print(f"  whale_activity: combined anomaly score")
    print(f"  tx_anomaly: transaction count anomaly")
    print(f"  miner_confidence: hashrate trend")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
