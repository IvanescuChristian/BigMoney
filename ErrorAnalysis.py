"""
ErrorAnalysis.py
────────────────
Compares predictions (predicted_hourly/) with reality (real_predictions/).
Run AFTER FetchRealData.py.

Outputs:
  error_analysis/
    ├── per_coin_report.csv      -- accuracy per coin
    ├── direction_report.csv     -- UP/DOWN/FLAT hit rate per coin
    ├── horizon_decay.csv        -- error growth over time
    ├── social_accuracy.csv      -- social volume prediction quality
    └── full_debug.csv           -- every hour, predicted vs real (for manual inspection)

Usage:  python ErrorAnalysis.py
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REAL_DIR   = os.path.join(BASE_DIR, "real_predictions")
PRED_DIR   = os.path.join(BASE_DIR, "predicted_hourly")
OUTPUT_DIR = os.path.join(BASE_DIR, "error_analysis")

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_real_data():
    """Load all real_predictions/*.csv (output of FetchRealData.py)."""
    files = sorted(glob.glob(os.path.join(REAL_DIR, "*.csv")))
    files = [f for f in files if "summary" not in os.path.basename(f).lower()]
    if not files:
        print("[ERROR] No files in real_predictions/. Run FetchRealData.py first.")
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['date_file'] = os.path.splitext(os.path.basename(f))[0]
            dfs.append(df)
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp_utc'] = pd.to_datetime(combined['timestamp_utc'])
    return combined


def load_prediction_summary():
    """Load prediction_summary.csv for start/end comparison."""
    path = os.path.join(PRED_DIR, "prediction_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  1. PER-COIN PRICE ACCURACY
# ═══════════════════════════════════════════════════════════════════════════

def analyze_price_accuracy(df):
    valid = df.dropna(subset=['real_price', 'predicted_price']).copy()
    if valid.empty:
        return pd.DataFrame()

    valid['error_usd'] = valid['real_price'] - valid['predicted_price']
    valid['error_pct'] = (valid['error_usd'] / valid['predicted_price']) * 100
    valid['abs_error_pct'] = valid['error_pct'].abs()

    rows = []
    for coin, g in valid.groupby('coin_id'):
        n = len(g)
        if n < 2:
            continue

        first_pred = g.iloc[0]['predicted_price']
        first_real = g.iloc[0]['real_price']
        last_pred = g.iloc[-1]['predicted_price']
        last_real = g.iloc[-1]['real_price']

        # Overall trend: did we predict the right direction over the full period?
        pred_trend = "UP" if last_pred > first_pred else "DOWN" if last_pred < first_pred else "FLAT"
        real_trend = "UP" if last_real > first_real else "DOWN" if last_real < first_real else "FLAT"
        trend_hit = pred_trend == real_trend

        rows.append({
            'coin_id': coin,
            'n_hours': n,
            'first_predicted': round(first_pred, 6),
            'first_real': round(first_real, 6),
            'last_predicted': round(last_pred, 6),
            'last_real': round(last_real, 6),
            'pred_trend': pred_trend,
            'real_trend': real_trend,
            'trend_correct': 'YES' if trend_hit else 'NO',
            'bias_pct': round(g['error_pct'].mean(), 4),
            'mae_pct': round(g['abs_error_pct'].mean(), 4),
            'max_error_pct': round(g['abs_error_pct'].max(), 4),
            'median_error_pct': round(g['abs_error_pct'].median(), 4),
        })

    return pd.DataFrame(rows).sort_values('mae_pct')


# ═══════════════════════════════════════════════════════════════════════════
#  2. DIRECTION ACCURACY (hourly UP/DOWN)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_direction(df):
    valid = df.dropna(subset=['real_price', 'predicted_price']).copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(['coin_id', 'timestamp_utc'])

    # Compute hourly changes
    valid['pred_change'] = valid.groupby('coin_id')['predicted_price'].diff()
    valid['real_change'] = valid.groupby('coin_id')['real_price'].diff()
    valid = valid.dropna(subset=['pred_change', 'real_change'])

    # Classify direction
    def direction(x):
        if x > 0: return 'UP'
        if x < 0: return 'DOWN'
        return 'FLAT'

    valid['pred_dir'] = valid['pred_change'].apply(direction)
    valid['real_dir'] = valid['real_change'].apply(direction)
    valid['dir_match'] = valid['pred_dir'] == valid['real_dir']

    rows = []
    for coin, g in valid.groupby('coin_id'):
        n = len(g)
        if n < 5:
            continue

        hits = g['dir_match'].sum()
        acc = hits / n

        # Break down by predicted direction
        pred_up = g[g['pred_dir'] == 'UP']
        pred_down = g[g['pred_dir'] == 'DOWN']

        up_correct = pred_up['dir_match'].sum() if len(pred_up) > 0 else 0
        down_correct = pred_down['dir_match'].sum() if len(pred_down) > 0 else 0

        rows.append({
            'coin_id': coin,
            'n_hours': n,
            'direction_accuracy': round(acc, 4),
            'direction_accuracy_pct': f"{acc*100:.1f}%",
            'total_hits': hits,
            'total_misses': n - hits,
            'pred_UP_count': len(pred_up),
            'pred_UP_correct': up_correct,
            'pred_UP_accuracy': f"{up_correct/len(pred_up)*100:.1f}%" if len(pred_up) > 0 else "N/A",
            'pred_DOWN_count': len(pred_down),
            'pred_DOWN_correct': down_correct,
            'pred_DOWN_accuracy': f"{down_correct/len(pred_down)*100:.1f}%" if len(pred_down) > 0 else "N/A",
        })

    return pd.DataFrame(rows).sort_values('direction_accuracy', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
#  3. SOCIAL VOLUME ACCURACY
# ═══════════════════════════════════════════════════════════════════════════

def analyze_social(df):
    valid = df.dropna(subset=['predicted_social', 'real_social']).copy()
    if valid.empty:
        print("  [INFO] No social volume comparison data available.")
        return pd.DataFrame()

    valid['social_error'] = valid['real_social'] - valid['predicted_social']
    valid['social_abs_error'] = valid['social_error'].abs()

    # For % error, handle zero predictions
    valid['social_error_pct'] = np.where(
        valid['predicted_social'] != 0,
        (valid['social_error'] / valid['predicted_social']) * 100,
        np.nan
    )
    valid['social_abs_error_pct'] = valid['social_error_pct'].abs()

    rows = []
    for coin, g in valid.groupby('coin_id'):
        n = len(g)
        if n < 2:
            continue

        avg_pred = g['predicted_social'].mean()
        avg_real = g['real_social'].mean()
        mae = g['social_abs_error'].mean()
        mae_pct_vals = g['social_abs_error_pct'].dropna()
        mae_pct = mae_pct_vals.mean() if len(mae_pct_vals) > 0 else np.nan

        rows.append({
            'coin_id': coin,
            'n_hours': n,
            'avg_predicted_social': round(avg_pred, 2),
            'avg_real_social': round(avg_real, 2),
            'social_mae': round(mae, 2),
            'social_mae_pct': round(mae_pct, 2) if not np.isnan(mae_pct) else 'N/A',
            'social_bias': round((avg_pred - avg_real), 2),
        })

    return pd.DataFrame(rows).sort_values('coin_id')


# ═══════════════════════════════════════════════════════════════════════════
#  4. HORIZON DECAY
# ═══════════════════════════════════════════════════════════════════════════

def analyze_horizon(df):
    valid = df.dropna(subset=['real_price', 'predicted_price']).copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(['coin_id', 'timestamp_utc'])
    valid['hour_ahead'] = valid.groupby('coin_id').cumcount() + 1
    valid['abs_error_pct'] = ((valid['real_price'] - valid['predicted_price']) / valid['predicted_price'] * 100).abs()
    valid['error_pct'] = (valid['real_price'] - valid['predicted_price']) / valid['predicted_price'] * 100

    # Buckets
    bins = [0, 6, 24, 48, 96, 168, 9999]
    labels = ['1-6h', '7-24h', '25-48h', '49-96h', '97-168h', '168h+']
    valid['bucket'] = pd.cut(valid['hour_ahead'], bins=bins, labels=labels)

    result = valid.groupby('bucket', observed=True).agg(
        mean_error_pct=('abs_error_pct', 'mean'),
        median_error_pct=('abs_error_pct', 'median'),
        mean_bias_pct=('error_pct', 'mean'),
        max_error_pct=('abs_error_pct', 'max'),
        n_points=('abs_error_pct', 'count')
    ).round(4).reset_index()

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  5. DEBUG FILE (every row, predicted vs real)
# ═══════════════════════════════════════════════════════════════════════════

def build_debug(df):
    valid = df.dropna(subset=['real_price', 'predicted_price']).copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(['coin_id', 'timestamp_utc'])
    valid['hour_ahead'] = valid.groupby('coin_id').cumcount() + 1
    valid['error_pct'] = ((valid['real_price'] - valid['predicted_price']) / valid['predicted_price'] * 100).round(4)
    valid['abs_error_pct'] = valid['error_pct'].abs()

    # Direction
    valid['pred_change'] = valid.groupby('coin_id')['predicted_price'].diff()
    valid['real_change'] = valid.groupby('coin_id')['real_price'].diff()
    valid['pred_dir'] = np.where(valid['pred_change'] > 0, 'UP', np.where(valid['pred_change'] < 0, 'DOWN', 'FLAT'))
    valid['real_dir'] = np.where(valid['real_change'] > 0, 'UP', np.where(valid['real_change'] < 0, 'DOWN', 'FLAT'))
    valid['dir_match'] = valid['pred_dir'] == valid['real_dir']

    cols = ['timestamp_utc', 'coin_id', 'hour_ahead',
            'predicted_price', 'real_price', 'error_pct', 'abs_error_pct',
            'pred_dir', 'real_dir', 'dir_match']

    # Add social if available
    if 'predicted_social' in valid.columns and 'real_social' in valid.columns:
        cols += ['predicted_social', 'real_social', 'social_diff_pct']

    return valid[[c for c in cols if c in valid.columns]]


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_report(price_df, dir_df, social_df, horizon_df, summary_df, raw_df):
    border = "=" * 72
    line   = "-" * 72

    print(f"\n{border}")
    print("  PREDICTION vs REALITY -- FULL ANALYSIS REPORT")
    print(f"{border}")

    # ── OVERALL NUMBERS ──────────────────────────────────────────────
    total_coins = 0
    total_hours = 0
    if not price_df.empty:
        total_coins = len(price_df)
        total_hours = price_df['n_hours'].sum()

    print(f"\n  Data: {total_coins} coins | {total_hours} total hour-predictions")

    # ── 1. PRICE ACCURACY ────────────────────────────────────────────
    if not price_df.empty:
        avg_mae = price_df['mae_pct'].mean()
        median_mae = price_df['mae_pct'].median()
        best = price_df.iloc[0]
        worst = price_df.iloc[-1]

        print(f"\n{line}")
        print(f"  1. PRICE ACCURACY")
        print(f"{line}")
        print(f"     Average MAE:  {avg_mae:.2f}%")
        print(f"     Median MAE:   {median_mae:.2f}%")
        print(f"     Best coin:    {best['coin_id']} (MAE {best['mae_pct']:.2f}%)")
        print(f"     Worst coin:   {worst['coin_id']} (MAE {worst['mae_pct']:.2f}%)")

        # Group by accuracy tier
        tier1 = price_df[price_df['mae_pct'] < 1]
        tier2 = price_df[(price_df['mae_pct'] >= 1) & (price_df['mae_pct'] < 5)]
        tier3 = price_df[(price_df['mae_pct'] >= 5) & (price_df['mae_pct'] < 10)]
        tier4 = price_df[price_df['mae_pct'] >= 10]

        print(f"\n     Accuracy tiers:")
        print(f"       < 1% error:   {len(tier1)} coins  ({len(tier1)/total_coins*100:.0f}%)")
        print(f"       1-5% error:   {len(tier2)} coins  ({len(tier2)/total_coins*100:.0f}%)")
        print(f"       5-10% error:  {len(tier3)} coins  ({len(tier3)/total_coins*100:.0f}%)")
        print(f"       > 10% error:  {len(tier4)} coins  ({len(tier4)/total_coins*100:.0f}%)")

        if len(tier4) > 0:
            print(f"\n     Worst offenders (>10% error):")
            for _, r in tier4.iterrows():
                print(f"       {r['coin_id']:<30} MAE={r['mae_pct']:.2f}%  Bias={r['bias_pct']:+.2f}%")

    # ── 2. DIRECTION ACCURACY ────────────────────────────────────────
    if not dir_df.empty:
        total_dir_hours = dir_df['n_hours'].sum()
        total_dir_hits = dir_df['total_hits'].sum()
        overall_dir_acc = total_dir_hits / total_dir_hours if total_dir_hours > 0 else 0

        print(f"\n{line}")
        print(f"  2. DIRECTION ACCURACY (hourly UP/DOWN)")
        print(f"{line}")
        print(f"     Overall: {total_dir_hits}/{total_dir_hours} = {overall_dir_acc*100:.1f}%")
        print(f"     (random would be ~33%, >50% is useful)")

        # Trend accuracy (overall period direction)
        if not price_df.empty:
            trend_correct = (price_df['trend_correct'] == 'YES').sum()
            print(f"\n     7-day trend direction (UP/DOWN over full period):")
            print(f"     Correct: {trend_correct}/{total_coins} = {trend_correct/total_coins*100:.1f}%")

        # Top/bottom 5
        print(f"\n     Top 5 directional coins:")
        for _, r in dir_df.head(5).iterrows():
            print(f"       {r['coin_id']:<30} {r['direction_accuracy_pct']}")

        print(f"\n     Bottom 5 directional coins:")
        for _, r in dir_df.tail(5).iterrows():
            print(f"       {r['coin_id']:<30} {r['direction_accuracy_pct']}")

    # ── 3. SOCIAL VOLUME ACCURACY ────────────────────────────────────
    if not social_df.empty:
        print(f"\n{line}")
        print(f"  3. SOCIAL VOLUME ACCURACY")
        print(f"{line}")
        print(f"     Coins with social data: {len(social_df)}")

        for _, r in social_df.iterrows():
            mae_str = f"{r['social_mae_pct']}%" if r['social_mae_pct'] != 'N/A' else 'N/A'
            print(f"       {r['coin_id']:<25} pred={r['avg_predicted_social']:<10} "
                  f"real={r['avg_real_social']:<10} MAE={mae_str}")
    else:
        print(f"\n{line}")
        print(f"  3. SOCIAL VOLUME ACCURACY")
        print(f"{line}")
        print(f"     No social comparison data. Run FetchRealData.py with Santiment.")

    # ── 4. ERROR vs HORIZON ──────────────────────────────────────────
    if not horizon_df.empty:
        print(f"\n{line}")
        print(f"  4. ERROR GROWTH OVER TIME")
        print(f"{line}")
        for _, r in horizon_df.iterrows():
            print(f"     {str(r['bucket']):<12} "
                  f"MAE={r['mean_error_pct']:.2f}%  "
                  f"Median={r['median_error_pct']:.2f}%  "
                  f"Bias={r['mean_bias_pct']:+.3f}%  "
                  f"Max={r['max_error_pct']:.2f}%  "
                  f"N={int(r['n_points'])}")

    # ── 5. PREDICTION SUMMARY vs REALITY ─────────────────────────────
    if summary_df is not None and not price_df.empty:
        print(f"\n{line}")
        print(f"  5. 7-DAY PREDICTION SUMMARY vs REALITY")
        print(f"{line}")
        print(f"     {'Coin':<28} {'Pred%':>8} {'Real%':>8} {'Trend':>8} {'MAE%':>8}")
        print(f"     {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        merged = summary_df.merge(price_df[['coin_id', 'first_real', 'last_real', 'trend_correct', 'mae_pct']],
                                   on='coin_id', how='left')

        for _, r in merged.iterrows():
            pred_chg = r.get('predicted_change_%', 0)
            if pd.notna(r.get('first_real')) and pd.notna(r.get('last_real')) and r['first_real'] != 0:
                real_chg = ((r['last_real'] - r['first_real']) / r['first_real']) * 100
            else:
                real_chg = float('nan')

            trend = r.get('trend_correct', '?')
            mae = r.get('mae_pct', float('nan'))

            real_str = f"{real_chg:+.2f}%" if not np.isnan(real_chg) else "N/A"
            mae_str = f"{mae:.2f}%" if not np.isnan(mae) else "N/A"

            print(f"     {r['coin_id']:<28} {pred_chg:>+7.2f}% {real_str:>8} {trend:>8} {mae_str:>8}")

    # ── 6. VERDICT ───────────────────────────────────────────────────
    print(f"\n{border}")
    print(f"  VERDICT")
    print(f"{border}")

    if not price_df.empty and not dir_df.empty:
        avg_mae = price_df['mae_pct'].mean()
        overall_dir_acc = dir_df['total_hits'].sum() / dir_df['n_hours'].sum() if dir_df['n_hours'].sum() > 0 else 0
        trend_acc = (price_df['trend_correct'] == 'YES').sum() / len(price_df) if len(price_df) > 0 else 0
        good_coins = len(price_df[price_df['mae_pct'] < 5])

        print(f"\n     Price MAE average:        {avg_mae:.2f}%")
        print(f"     Hourly direction accuracy: {overall_dir_acc*100:.1f}%")
        print(f"     7-day trend accuracy:      {trend_acc*100:.1f}%")
        print(f"     Coins under 5% error:      {good_coins}/{total_coins}")

        score = 0
        if avg_mae < 5: score += 1
        if overall_dir_acc > 0.45: score += 1
        if trend_acc > 0.55: score += 1
        if good_coins > total_coins * 0.6: score += 1

        print(f"\n     TRADING READINESS: {score}/4")
        if score >= 3:
            print("     -> CAUTIOUS GO -- apply correction factors, paper trade first")
        elif score >= 2:
            print("     -> NEEDS WORK -- model shows promise but needs tuning")
        else:
            print("     -> NOT READY -- significant improvements needed")

    print(f"\n  Files saved in: {OUTPUT_DIR}/")
    print(f"{border}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(">>> Loading comparison data from real_predictions/...")
    df = load_real_data()
    if df is None:
        return

    valid_prices = df.dropna(subset=['real_price', 'predicted_price'])
    print(f"    Loaded {len(df)} total rows, {len(valid_prices)} with real price data")
    print(f"    Coins: {df['coin_id'].nunique()}")
    print(f"    Dates: {sorted(df['date_file'].unique())}\n")

    summary_df = load_prediction_summary()

    # 1. Price accuracy
    print(">>> [1/5] Analyzing price accuracy...")
    price_report = analyze_price_accuracy(df)
    if not price_report.empty:
        price_report.to_csv(os.path.join(OUTPUT_DIR, "per_coin_report.csv"), index=False)

    # 2. Direction accuracy
    print(">>> [2/5] Analyzing direction accuracy...")
    dir_report = analyze_direction(df)
    if not dir_report.empty:
        dir_report.to_csv(os.path.join(OUTPUT_DIR, "direction_report.csv"), index=False)

    # 3. Social accuracy
    print(">>> [3/5] Analyzing social volume accuracy...")
    social_report = analyze_social(df)
    if not social_report.empty:
        social_report.to_csv(os.path.join(OUTPUT_DIR, "social_accuracy.csv"), index=False)

    # 4. Horizon decay
    print(">>> [4/5] Analyzing error vs horizon...")
    horizon_report = analyze_horizon(df)
    if not horizon_report.empty:
        horizon_report.to_csv(os.path.join(OUTPUT_DIR, "horizon_decay.csv"), index=False)

    # 5. Debug file
    print(">>> [5/5] Building debug file...")
    debug_df = build_debug(df)
    if not debug_df.empty:
        debug_df.to_csv(os.path.join(OUTPUT_DIR, "full_debug.csv"), index=False)
        print(f"    Debug file: {len(debug_df)} rows")

    # Report
    print_report(price_report, dir_report, social_report, horizon_report, summary_df, df)


if __name__ == "__main__":
    run()
