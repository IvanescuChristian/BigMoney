"""
rebuild_comparison.py
─────────────────────
Rebuilds real_predictions/ comparison files using:
  - NEW predicted prices from predicted_hourly/
  - EXISTING real prices from real_predictions/ (already fetched)

This avoids re-running FetchRealData.py (which is slow with proxies).
"""

import pandas as pd
import numpy as np
import os
import glob

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PRED_DIR    = os.path.join(BASE_DIR, "predicted_hourly")
REAL_DIR    = os.path.join(BASE_DIR, "real_predictions")

def run():
    print("=" * 60)
    print("  REBUILD COMPARISON — reuse real prices, update predictions")
    print("=" * 60)

    # 1. Load existing real prices from real_predictions/
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.csv")))
    real_files = [f for f in real_files if "summary" not in os.path.basename(f).lower()]

    if not real_files:
        print("[ERROR] No files in real_predictions/. Run FetchRealData.py first.")
        return

    # Extract real prices: {date_str: {coin_id: {hour_index: real_price}}}
    real_prices = {}
    for fp in real_files:
        date_str = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_csv(fp)
            if 'real_price' not in df.columns:
                print(f"  [SKIP] {date_str} — no real_price column")
                continue
            real_prices[date_str] = df[['timestamp_utc', 'coin_id', 'real_price',
                                         'real_social']].copy() if 'real_social' in df.columns \
                                   else df[['timestamp_utc', 'coin_id', 'real_price']].copy()
            print(f"  [REAL] {date_str} — {len(df)} rows, "
                  f"{df['real_price'].notna().sum()} with real price")
        except Exception as e:
            print(f"  [SKIP] {date_str}: {e}")

    if not real_prices:
        print("[ERROR] No real price data found.")
        return

    # 2. Load NEW predictions from predicted_hourly/
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")))
    pred_files = [f for f in pred_files if "summary" not in os.path.basename(f).lower()]

    if not pred_files:
        print("[ERROR] No prediction files in predicted_hourly/")
        return

    print(f"\n  Rebuilding {len(pred_files)} comparison files...\n")

    for fp in pred_files:
        date_str = os.path.splitext(os.path.basename(fp))[0]

        if date_str not in real_prices:
            print(f"  [SKIP] {date_str} — no real data for this date")
            continue

        try:
            pred_df = pd.read_csv(fp)
            real_df = real_prices[date_str]

            # Round both to nearest hour to ensure matching
            pred_df['timestamp_utc'] = pd.to_datetime(pred_df['timestamp_utc']).dt.round('h')
            real_df = real_df.copy()
            real_df['timestamp_utc'] = pd.to_datetime(real_df['timestamp_utc']).dt.round('h')

            # Drop duplicates after rounding (keep last)
            pred_df = pred_df.drop_duplicates(subset=['timestamp_utc', 'coin_id'], keep='last')
            real_df = real_df.drop_duplicates(subset=['timestamp_utc', 'coin_id'], keep='last')

            merged = pred_df.merge(real_df, on=['timestamp_utc', 'coin_id'], how='left')

            # Build comparison columns
            rows = []
            for _, row in merged.iterrows():
                pred_price = row['price_usd']
                real_price = row.get('real_price', np.nan)
                pred_social = row.get('predicted_social_volume', np.nan)
                real_social = row.get('real_social', np.nan)
                pred_error = row.get('error', np.nan)

                p_diff = (real_price - pred_price) if pd.notna(real_price) else np.nan
                p_diff_pct = (p_diff / pred_price * 100) if (pd.notna(p_diff) and pred_price != 0) else np.nan

                s_diff = np.nan
                s_diff_pct = np.nan
                if pd.notna(real_social) and pd.notna(pred_social):
                    s_diff = real_social - pred_social
                    s_diff_pct = (s_diff / pred_social * 100) if pred_social != 0 else np.nan

                rows.append({
                    'timestamp_utc': row['timestamp_utc'],
                    'coin_id': row['coin_id'],
                    'predicted_price': round(pred_price, 6),
                    'real_price': round(real_price, 6) if pd.notna(real_price) else np.nan,
                    'price_diff': round(p_diff, 6) if pd.notna(p_diff) else np.nan,
                    'price_diff_pct': round(p_diff_pct, 4) if pd.notna(p_diff_pct) else np.nan,
                    'predicted_social': round(pred_social, 2) if pd.notna(pred_social) else np.nan,
                    'real_social': round(real_social, 2) if pd.notna(real_social) else np.nan,
                    'social_diff': round(s_diff, 2) if pd.notna(s_diff) else np.nan,
                    'social_diff_pct': round(s_diff_pct, 2) if pd.notna(s_diff_pct) else np.nan,
                    'pred_std_error': round(pred_error, 6) if pd.notna(pred_error) else np.nan
                })

            out_df = pd.DataFrame(rows)
            out_path = os.path.join(REAL_DIR, f"{date_str}.csv")
            out_df.to_csv(out_path, index=False)

            ok_p = out_df['real_price'].notna().sum()
            ok_s = out_df['real_social'].notna().sum()
            print(f"  [OK] {date_str} — {len(out_df)} rows | price:{ok_p} social:{ok_s}")

        except Exception as e:
            print(f"  [ERROR] {date_str}: {e}")

    # Rebuild summary
    all_csvs = [f for f in glob.glob(os.path.join(REAL_DIR, "*.csv"))
                if "summary" not in os.path.basename(f).lower()]
    if all_csvs:
        combined = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        valid = combined.dropna(subset=['real_price', 'predicted_price'])
        if not valid.empty:
            summary = valid.groupby('coin_id').agg(
                avg_predicted=('predicted_price', 'mean'),
                avg_real=('real_price', 'mean'),
                avg_error_pct=('price_diff_pct', 'mean'),
                max_error_pct=('price_diff_pct', lambda x: x.abs().max()),
                num_hours=('coin_id', 'count')
            ).round(4).reset_index()
            summary.to_csv(os.path.join(REAL_DIR, "comparison_summary.csv"), index=False)
            print(f"\n  [SUMMARY] comparison_summary.csv ({len(summary)} coins)")

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
