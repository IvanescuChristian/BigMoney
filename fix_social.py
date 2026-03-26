"""
fix_social.py
─────────────
Fetches social_volume for ALL days in historical_hourly/ using ONE request
per coin (full date range), then distributes values across CSV files.

This uses ~45 API calls total instead of 45 × 55 = 2475.
"""

import pandas as pd
import san
import os
import sys
import time
import glob
from datetime import datetime, timedelta

san.ApiConfig.api_key = 'yvib2jijbrcivyfh_uw25dl5mk4qch5mf'

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HIST_DIR  = os.path.join(BASE_DIR, "historical_hourly")

# Verified working Santiment slugs
COINGECKO_TO_SANTIMENT = {
    "bitcoin":           "bitcoin",
    "ethereum":          "ethereum",
    "tether":            "tether",
    "ripple":            "ripple",
    "binancecoin":       "binance-coin",
    "usd-coin":          "usd-coin",
    "solana":            "solana",
    "tron":              "tron",
    "dogecoin":          "dogecoin",
    "cardano":           "cardano",
    "bitcoin-cash":      "bitcoin-cash",
    "hyperliquid":       "hyperliquid",
    "leo-token":         "unus-sed-leo",
    "chainlink":         "chainlink",
    "monero":            "monero",
    "stellar":           "stellar",
    "dai":               "multi-collateral-dai",
    "litecoin":          "litecoin",
    "avalanche-2":       "avalanche",
    "hedera-hashgraph":  "hedera-hashgraph",
    "sui":               "sui",
    "shiba-inu":         "shiba-inu",
    "polkadot":          "polkadot-new",
    "uniswap":           "uniswap",
    "near":              "near-protocol",
    "aave":              "aave",
    "bittensor":         "bittensor",
    "okb":               "okb",
    "zcash":             "zcash",
    "ethereum-classic":  "ethereum-classic",
    "mantle":            "mantle",
    "pax-gold":          "pax-gold",
    "tether-gold":       "tether-gold",
    "internet-computer": "internet-computer",
    "crypto-com-chain":  "crypto-com-coin",
    "paypal-usd":        "paypal-usd",
    "kaspa":             "kaspa",
    "pepe":              "pepe",
    "aptos":             "aptos",
    "ondo-finance":      "ondo-finance",
    "ethena-usde":       "ethena-usde",
    "jito-staked-sol":   "jito-staked-sol",
    "usds":              "usds",
    "weth":              "weth",
    "wrapped-eeth":      "wrapped-eeth",
}

RATE_LIMIT_WAIT = 1860  # 31 min


def fetch_social_range(slug, start_date, end_date):
    """Fetch social_volume for entire date range in ONE call.
    Returns dict: {date_str: value} or None on failure."""
    s_date = f"{start_date}T00:00:00Z"
    e_date = f"{end_date}T23:59:59Z"

    try:
        data = san.get(f"social_volume_total/{slug}",
                       s_date=s_date, e_date=e_date, interval="1d")
        if data.empty or 'value' not in data.columns:
            return {}

        result = {}
        for idx, row in data.iterrows():
            # Index is datetime
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
            result[date_str] = float(row['value'])
        return result

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "Rate Limit" in err_str:
            return "RATE_LIMITED"
        print(f"    [ERROR] {slug}: {err_str[:120]}")
        return {}


def main():
    print("=" * 60)
    print("  FIX SOCIAL DATA -- Bulk fetch (1 request per coin)")
    print("=" * 60)

    # Find date range from CSV files
    csv_files = sorted(glob.glob(os.path.join(HIST_DIR, "*.csv")))
    if not csv_files:
        print("No CSV files in historical_hourly/")
        return

    dates = []
    for f in csv_files:
        bn = os.path.splitext(os.path.basename(f))[0]
        try:
            dates.append(bn)
        except:
            pass

    dates.sort()
    start_date = dates[0]
    end_date = dates[-1]
    print(f"\n  Date range: {start_date} -> {end_date} ({len(dates)} days)")

    # Find all coins across all files
    all_coins = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=['coin_id'])
            all_coins.update(df['coin_id'].unique())
        except:
            pass

    coins_with_slugs = {c: s for c, s in COINGECKO_TO_SANTIMENT.items() if c in all_coins}
    print(f"  Coins with Santiment slugs: {len(coins_with_slugs)}/{len(all_coins)}")
    print(f"  API calls needed: {len(coins_with_slugs)} (one per coin)\n")

    # Fetch social data: coin -> {date -> value}
    social_data = {}  # coin_id -> {date_str -> sv}

    for i, (coin_id, slug) in enumerate(sorted(coins_with_slugs.items()), 1):
        print(f"[{i}/{len(coins_with_slugs)}] {coin_id} -> {slug}...", end="", flush=True)

        result = fetch_social_range(slug, start_date, end_date)

        if result == "RATE_LIMITED":
            print(f" [RATE LIMITED]")
            print(f"\n  Waiting {RATE_LIMIT_WAIT // 60} minutes for rate limit reset...")
            print(f"  Resume at: {datetime.now() + timedelta(seconds=RATE_LIMIT_WAIT):%H:%M:%S}")
            time.sleep(RATE_LIMIT_WAIT)

            # Retry
            result = fetch_social_range(slug, start_date, end_date)
            if result == "RATE_LIMITED":
                print(f"  Still limited. Saving what we have.")
                break

        if isinstance(result, dict) and result:
            social_data[coin_id] = result
            # Show a sample
            sample_val = list(result.values())[0]
            print(f" OK ({len(result)} days, sample={sample_val})")
        else:
            print(f" [NO DATA]")

        time.sleep(1.5)

    if not social_data:
        print("\nNo social data fetched.")
        return

    # Write to CSV files
    print(f"\n  Writing social data to {len(dates)} CSV files...")
    updated = 0

    for date_str in dates:
        csv_path = os.path.join(HIST_DIR, f"{date_str}.csv")
        try:
            df = pd.read_csv(csv_path)
        except:
            continue

        sv_col = f"social_volume_{date_str.replace('-', '_')}"

        # Map coin_id -> social value for this date
        sv_values = {}
        for coin_id, date_map in social_data.items():
            if date_str in date_map:
                sv_values[coin_id] = date_map[date_str]

        if sv_values:
            df[sv_col] = df['coin_id'].map(sv_values)
            df.to_csv(csv_path, index=False)
            updated += 1

    print(f"  Updated {updated}/{len(dates)} CSV files")
    print(f"  Coins with data: {len(social_data)}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE -- {len(social_data)} coins x {len(dates)} days")
    print(f"  API calls used: {len(coins_with_slugs)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
