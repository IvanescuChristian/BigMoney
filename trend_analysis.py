"""
trend_analysis.py
─────────────────
Deep analysis of trend prediction quality.
Answers: when we say UP, how often is it UP? When we say -5%, how close to -5% is reality?
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "predicted_hourly")
REAL_DIR = os.path.join(BASE_DIR, "real_predictions")
OUT_DIR  = os.path.join(BASE_DIR, "error_analysis")


def load_summary():
    """Load prediction summary and match with real outcomes."""
    summary_path = os.path.join(PRED_DIR, "prediction_summary.csv")
    if not os.path.exists(summary_path):
        print("No prediction_summary.csv found")
        return None

    pred = pd.read_csv(summary_path)

    # Load real data from comparison files
    import glob
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.csv")))
    real_files = [f for f in real_files if "summary" not in os.path.basename(f).lower()]

    if not real_files:
        print("No real comparison files")
        return None

    # Get first and last real price per coin
    all_real = pd.concat([pd.read_csv(f) for f in real_files], ignore_index=True)
    all_real = all_real.dropna(subset=['real_price'])
    all_real['timestamp_utc'] = pd.to_datetime(all_real['timestamp_utc'])

    coin_real = []
    for coin, grp in all_real.groupby('coin_id'):
        grp = grp.sort_values('timestamp_utc')
        first_price = grp['real_price'].iloc[0]
        last_price = grp['real_price'].iloc[-1]
        real_pct = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0
        coin_real.append({
            'coin_id': coin,
            'real_start': first_price,
            'real_end': last_price,
            'real_pct': real_pct,
            'real_direction': 'UP' if real_pct > 0.1 else ('DOWN' if real_pct < -0.1 else 'FLAT'),
            'n_hours': len(grp)
        })

    real_df = pd.DataFrame(coin_real)
    merged = pred.merge(real_df, on='coin_id', how='inner')
    merged['pred_pct'] = merged['predicted_change_%']
    merged['pred_direction'] = merged['pred_pct'].apply(
        lambda x: 'UP' if x > 0.1 else ('DOWN' if x < -0.1 else 'FLAT'))

    return merged


def analyze_trends(df):
    print("=" * 70)
    print("  TREND PREDICTION QUALITY ANALYSIS")
    print("=" * 70)

    total = len(df)

    # 1. Direction accuracy
    correct = (df['pred_direction'] == df['real_direction']).sum()
    print(f"\n  Overall direction: {correct}/{total} = {correct/total*100:.1f}%\n")

    # 2. Confusion matrix
    print("  CONFUSION MATRIX:")
    print("  " + "-" * 50)
    directions = ['UP', 'DOWN', 'FLAT']
    for pred_dir in directions:
        for real_dir in directions:
            count = ((df['pred_direction'] == pred_dir) & (df['real_direction'] == real_dir)).sum()
            if count > 0:
                pct = count / total * 100
                marker = " <--" if pred_dir == real_dir else ""
                print(f"    Pred {pred_dir:>4} / Real {real_dir:>4}: {count:>3} ({pct:>5.1f}%){marker}")
    print()

    # 3. Precision & Recall per direction
    print("  PRECISION & RECALL:")
    print("  " + "-" * 50)
    for direction in ['UP', 'DOWN']:
        pred_count = (df['pred_direction'] == direction).sum()
        real_count = (df['real_direction'] == direction).sum()
        tp = ((df['pred_direction'] == direction) & (df['real_direction'] == direction)).sum()

        precision = tp / pred_count * 100 if pred_count > 0 else 0
        recall = tp / real_count * 100 if real_count > 0 else 0

        print(f"    {direction}:")
        print(f"      Predicted {direction}: {pred_count} times")
        print(f"      Actually {direction}:  {real_count} times")
        print(f"      Correct:          {tp}")
        print(f"      Precision:        {precision:.1f}% (when we say {direction}, how often right)")
        print(f"      Recall:           {recall:.1f}% (when it IS {direction}, did we catch it)")
        print()

    # 4. Magnitude accuracy — when direction is correct
    print("  MAGNITUDE ACCURACY (direction-correct coins only):")
    print("  " + "-" * 50)
    correct_mask = df['pred_direction'] == df['real_direction']
    correct_df = df[correct_mask].copy()

    if len(correct_df) > 0:
        correct_df['mag_error'] = (correct_df['pred_pct'] - correct_df['real_pct']).abs()
        correct_df['mag_ratio'] = np.where(
            correct_df['real_pct'].abs() > 0.1,
            correct_df['pred_pct'] / correct_df['real_pct'],
            np.nan
        )

        avg_mag_error = correct_df['mag_error'].mean()
        med_mag_error = correct_df['mag_error'].median()

        print(f"    Correct predictions: {len(correct_df)}")
        print(f"    Avg magnitude error: {avg_mag_error:.2f}%")
        print(f"    Median magnitude error: {med_mag_error:.2f}%")
        print()

        # Categorize magnitude accuracy
        underpredict = correct_df[correct_df['mag_ratio'].notna() & (correct_df['mag_ratio'].abs() < 0.5)]
        close = correct_df[correct_df['mag_ratio'].notna() &
                           (correct_df['mag_ratio'].abs() >= 0.5) &
                           (correct_df['mag_ratio'].abs() <= 2.0)]
        overpredict = correct_df[correct_df['mag_ratio'].notna() & (correct_df['mag_ratio'].abs() > 2.0)]

        print(f"    Magnitude within 2x:   {len(close):>3} coins (good)")
        print(f"    Under-predicts (>2x):  {len(underpredict):>3} coins (says -2% real is -10%)")
        print(f"    Over-predicts (>2x):   {len(overpredict):>3} coins (says -20% real is -2%)")
        print()

        # Show detailed
        print("    Direction-correct coins detail:")
        print(f"    {'Coin':<35} {'Pred%':>8} {'Real%':>8} {'Ratio':>8}")
        print(f"    {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in correct_df.sort_values('mag_error').iterrows():
            ratio_str = f"{row['mag_ratio']:.2f}" if pd.notna(row['mag_ratio']) else "FLAT"
            print(f"    {row['coin_id']:<35} {row['pred_pct']:>+7.2f}% {row['real_pct']:>+7.2f}% {ratio_str:>8}")

    # 5. Wrong direction analysis
    print(f"\n  WRONG DIRECTION CALLS:")
    print("  " + "-" * 50)
    wrong_df = df[~correct_mask].copy()
    wrong_df['missed_by'] = (wrong_df['pred_pct'] - wrong_df['real_pct']).abs()

    print(f"    Total wrong: {len(wrong_df)}")
    if len(wrong_df) > 0:
        print(f"\n    {'Coin':<35} {'Pred%':>8} {'Real%':>8} {'Off by':>8}")
        print(f"    {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in wrong_df.sort_values('missed_by', ascending=False).iterrows():
            print(f"    {row['coin_id']:<35} {row['pred_pct']:>+7.2f}% {row['real_pct']:>+7.2f}% {row['missed_by']:>7.2f}%")

    # 6. Confidence score — coins where model has strong signal
    print(f"\n  CONFIDENCE TIERS:")
    print("  " + "-" * 50)

    df['abs_pred'] = df['pred_pct'].abs()
    for tier_name, min_pred, max_pred in [
        ("High confidence (>5% predicted move)", 5, 999),
        ("Medium confidence (1-5%)", 1, 5),
        ("Low confidence (<1%)", 0, 1)
    ]:
        tier = df[(df['abs_pred'] >= min_pred) & (df['abs_pred'] < max_pred)]
        if len(tier) > 0:
            tier_correct = (tier['pred_direction'] == tier['real_direction']).sum()
            print(f"    {tier_name}:")
            print(f"      Coins: {len(tier)}, Correct: {tier_correct}/{len(tier)} "
                  f"= {tier_correct/len(tier)*100:.1f}%")

    # 7. Save report
    report_path = os.path.join(OUT_DIR, "trend_quality.csv")
    df_out = df[['coin_id', 'pred_pct', 'real_pct', 'pred_direction', 'real_direction',
                  'n_hours']].copy()
    df_out['direction_correct'] = df_out['pred_direction'] == df_out['real_direction']
    df_out['magnitude_error'] = (df_out['pred_pct'] - df_out['real_pct']).abs()
    df_out.to_csv(report_path, index=False)
    print(f"\n  Saved: {report_path}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    df = load_summary()
    if df is not None:
        analyze_trends(df)
