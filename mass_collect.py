"""
mass_collect.py
───────────────
Collects historical hourly data for ALL missing dates.
Calls FetchPrevData.py one day at a time with cursor save.

Features:
  - Reads cursor from collect_cursor.txt (resumes where it left off)
  - Skips dates that already have CSVs in historical_hourly/
  - Saves cursor after EVERY completed day (safe to Ctrl+C anytime)
  - After all price data: runs FillData.py to add indicators
  - After FillData: runs whale_collector.py to refresh whale metrics

Usage:
    python mass_collect.py                     # from cursor or 2025-05-25
    python mass_collect.py --from 2025-06-01   # override start date
    python mass_collect.py --status            # show progress without fetching
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
HIST_DIR    = os.path.join(BASE_DIR, "historical_hourly")
CURSOR_FILE = os.path.join(BASE_DIR, "collect_cursor.txt")
DEFAULT_START = "2025-05-25"


def read_cursor():
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE, 'r') as f:
            d = f.read().strip()
            if d:
                try:
                    datetime.strptime(d, '%Y-%m-%d')
                    return d
                except ValueError:
                    pass
    return DEFAULT_START


def save_cursor(date_str):
    with open(CURSOR_FILE, 'w') as f:
        f.write(date_str)


def date_range(start_str, end_str):
    """Generate list of date strings from start to end (inclusive)."""
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    return dates


def csv_exists(date_str):
    path = os.path.join(HIST_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return False
    # Check it's not empty (at least header + 1 row)
    try:
        size = os.path.getsize(path)
        return size > 100  # empty CSV with header is ~50 bytes
    except:
        return False


def show_status():
    """Show collection progress."""
    cursor = read_cursor()
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    all_dates = date_range(DEFAULT_START, today)

    existing = [d for d in all_dates if csv_exists(d)]
    missing = [d for d in all_dates if not csv_exists(d)]

    print(f"\n  Cursor: {cursor}")
    print(f"  Range: {DEFAULT_START} -> {today} ({len(all_dates)} days)")
    print(f"  Collected: {len(existing)} days")
    print(f"  Missing: {len(missing)} days")

    if missing:
        print(f"\n  First missing: {missing[0]}")
        print(f"  Last missing:  {missing[-1]}")

        # Show gaps
        if len(missing) <= 20:
            print(f"\n  Missing dates:")
            for d in missing:
                print(f"    {d}")
    print()


def main():
    print("=" * 60)
    print("  MASS COLLECT -- Historical Data Collection")
    print("  One day at a time with cursor save")
    print("=" * 60)

    if '--status' in sys.argv:
        show_status()
        return

    # Determine start date
    if '--from' in sys.argv:
        idx = sys.argv.index('--from')
        start_date = sys.argv[idx + 1]
        save_cursor(start_date)
    else:
        start_date = read_cursor()

    # End date: yesterday (today's data may be incomplete)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')

    all_dates = date_range(start_date, yesterday)

    # Filter out existing
    missing = [d for d in all_dates if not csv_exists(d)]

    print(f"\n  Cursor: {start_date}")
    print(f"  End: {yesterday}")
    print(f"  Total range: {len(all_dates)} days")
    print(f"  Already have: {len(all_dates) - len(missing)} days")
    print(f"  To fetch: {len(missing)} days")

    if not missing:
        print("\n  All dates already collected!")
        print("  Running FillData.py to ensure indicators are up to date...")
        subprocess.run([sys.executable, "FillData.py"], cwd=BASE_DIR)
        print("\n  Running whale_collector.py to refresh whale metrics...")
        subprocess.run([sys.executable, "whale_collector.py"], cwd=BASE_DIR)
        return

    print(f"\n  Starting collection...\n")
    print(f"  TIP: Safe to Ctrl+C anytime. Cursor saves after each day.")
    print(f"  TIP: Restart with: python mass_collect.py\n")

    completed = 0
    failed_dates = []

    for i, date_str in enumerate(missing, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(missing)}] Collecting {date_str}...")
        print(f"  Cursor will save after completion")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                [sys.executable, "FetchPrevData.py", date_str],
                cwd=BASE_DIR,
                timeout=600  # 10 min max per day
            )

            if result.returncode == 0 and csv_exists(date_str):
                completed += 1
                # Save cursor to NEXT date (so we don't re-fetch this one)
                next_date = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                save_cursor(next_date)
                print(f"\n  [OK] {date_str} complete. Cursor saved: {next_date}")
                print(f"  Progress: {completed}/{len(missing)} days")
            else:
                print(f"\n  [WARN] {date_str} may have failed (no CSV or non-zero exit)")
                failed_dates.append(date_str)
                # Still advance cursor past this date
                next_date = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                save_cursor(next_date)

        except subprocess.TimeoutExpired:
            print(f"\n  [TIMEOUT] {date_str} took >10 min, skipping")
            failed_dates.append(date_str)
            next_date = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            save_cursor(next_date)

        except KeyboardInterrupt:
            print(f"\n\n  [STOPPED] Interrupted by user at {date_str}")
            print(f"  Cursor: {read_cursor()}")
            print(f"  Completed: {completed} days this session")
            print(f"  Resume with: python mass_collect.py")
            return

    # Summary
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Fetched: {completed}/{len(missing)} days")
    if failed_dates:
        print(f"  Failed: {len(failed_dates)} days: {', '.join(failed_dates[:10])}")
    print(f"{'='*60}")

    # Post-processing
    print(f"\n  Running FillData.py (add RSI, volatility, social)...")
    subprocess.run([sys.executable, "FillData.py"], cwd=BASE_DIR)

    print(f"\n  Running whale_collector.py (refresh whale metrics)...")
    subprocess.run([sys.executable, "whale_collector.py"], cwd=BASE_DIR)

    print(f"\n{'='*60}")
    print(f"  ALL DONE!")
    print(f"  historical_hourly/ now has data for prediction training")
    print(f"  Next: python Predict.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
