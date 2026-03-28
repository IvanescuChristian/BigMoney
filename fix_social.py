"""
fix_social.py
─────────────
Fetches social_volume for ALL days in historical_hourly/ using ONE request
per coin (full date range). 45 API calls total.

Features:
  - Rotates through up to 5 Santiment API keys on rate limit
  - Saves progress after EACH coin (cursor file)
  - Can resume if interrupted (skips already-fetched coins)
  - Validates data before writing

Usage:
    python fix_social.py              # normal run
    python fix_social.py --fresh      # ignore cursor, refetch all
"""

import pandas as pd
import san
import os
import sys
import time
import glob
import json
from datetime import datetime, timedelta

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HIST_DIR  = os.path.join(BASE_DIR, "historical_hourly")
CURSOR_FILE = os.path.join(BASE_DIR, "social_cursor.json")

# ── API Key Management ──────────────────────────────────────────────────────

def load_api_keys():
    """Load all Santiment API keys from .env. Skips placeholders."""
    keys = []
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        print(f"  [ERROR] .env file not found at {env_path}")
        return keys
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.upper().startswith('SANTIMENT_API_KEY'):
                parts = line.split('=', 1)
                if len(parts) != 2:
                    continue
                val = parts[1].strip()
                # Skip ALL placeholder patterns
                if not val or 'PUT_YOUR' in val.upper() or 'YOUR_KEY' in val.lower() \
                   or val == 'your_key_here' or len(val) < 10:
                    continue
                keys.append(val)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


API_KEYS = load_api_keys()
current_key_idx = 0


def set_api_key(idx):
    global current_key_idx
    if not API_KEYS:
        return
    current_key_idx = idx % len(API_KEYS)
    san.ApiConfig.api_key = API_KEYS[current_key_idx]


def try_all_keys_for_coin(slug, start_date, end_date):
    """
    Try fetching social data for a coin, rotating through ALL keys if rate limited.
    Returns dict {date: value}, empty dict on no data, or "ALL_LIMITED" string.
    """
    global current_key_idx
    tried_keys = set()

    while len(tried_keys) < len(API_KEYS):
        set_api_key(current_key_idx)
        tried_keys.add(current_key_idx)

        result = _fetch_one(slug, start_date, end_date)

        if result == "RATE_LIMITED":
            print(f" [key #{current_key_idx+1} limited]", end="", flush=True)
            # Try next key
            next_idx = (current_key_idx + 1) % len(API_KEYS)
            if next_idx in tried_keys:
                # We've tried all keys
                break
            current_key_idx = next_idx
            time.sleep(2)
            continue

        return result  # dict or empty dict

    return "ALL_LIMITED"


def _fetch_one(slug, start_date, end_date):
    """Single fetch attempt with current key. Returns dict, empty dict, or "RATE_LIMITED"."""
    s_date = f"{start_date}T00:00:00Z"
    e_date = f"{end_date}T23:59:59Z"

    try:
        data = san.get(f"social_volume_total/{slug}",
                       s_date=s_date, e_date=e_date, interval="1d")

        if data is None or data.empty or 'value' not in data.columns:
            return {}

        result = {}
        for idx, row in data.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
            val = float(row['value'])
            if val >= 0:  # sanity check
                result[date_str] = val
        return result

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "Rate Limit" in err_str or "rate limit" in err_str.lower():
            return "RATE_LIMITED"
        print(f" [ERROR: {err_str[:80]}]", end="")
        return {}


# ── Cursor (resume support) ─────────────────────────────────────────────────

def load_cursor():
    """Load progress: which coins already have social data fetched."""
    if os.path.exists(CURSOR_FILE):
        try:
            with open(CURSOR_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"completed_coins": [], "social_data": {}}


def save_cursor(cursor):
    with open(CURSOR_FILE, 'w') as f:
        json.dump(cursor, f, indent=2)


# ── Verified Santiment slugs ────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIX SOCIAL DATA -- Bulk fetch (1 request per coin)")
    print("  With key rotation + cursor resume")
    print("=" * 60)

    if not API_KEYS:
        print("\n  [ERROR] No valid Santiment API keys in .env")
        print("  Add keys like:")
        print("    SANTIMENT_API_KEY=your_real_key_1")
        print("    SANTIMENT_API_KEY_2=your_real_key_2")
        return

    print(f"\n  API keys loaded: {len(API_KEYS)}")
    for i, k in enumerate(API_KEYS):
        print(f"    Key #{i+1}: {k[:8]}...{k[-4:]}")

    # Find date range
    csv_files = sorted(glob.glob(os.path.join(HIST_DIR, "*.csv")))
    if not csv_files:
        print("  No CSV files in historical_hourly/")
        return

    dates = []
    for f in csv_files:
        bn = os.path.splitext(os.path.basename(f))[0]
        try:
            datetime.strptime(bn, '%Y-%m-%d')
            dates.append(bn)
        except ValueError:
            pass

    dates.sort()
    start_date = dates[0]
    end_date = dates[-1]
    print(f"\n  Date range: {start_date} -> {end_date} ({len(dates)} days)")

    # Find coins
    all_coins = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=['coin_id'])
            all_coins.update(df['coin_id'].unique())
        except:
            pass

    coins_with_slugs = {c: s for c, s in COINGECKO_TO_SANTIMENT.items() if c in all_coins}
    print(f"  Coins with Santiment slugs: {len(coins_with_slugs)}/{len(all_coins)}")

    # Load cursor (resume support)
    fresh = '--fresh' in sys.argv
    if fresh:
        cursor = {"completed_coins": [], "social_data": {}}
        print(f"\n  [FRESH] Ignoring previous progress")
    else:
        cursor = load_cursor()
        if cursor["completed_coins"]:
            print(f"\n  Resuming: {len(cursor['completed_coins'])} coins already done")

    completed = set(cursor["completed_coins"])
    social_data = cursor["social_data"]  # coin_id -> {date -> value}

    # Determine what to fetch
    to_fetch = {c: s for c, s in sorted(coins_with_slugs.items()) if c not in completed}
    print(f"  To fetch: {len(to_fetch)} coins")
    print(f"  API calls needed: ~{len(to_fetch)} (1 per coin)\n")

    if not to_fetch:
        print("  All coins already fetched! Writing to CSVs...")
    else:
        set_api_key(0)

    # Fetch
    for i, (coin_id, slug) in enumerate(to_fetch.items(), 1):
        print(f"  [{i}/{len(to_fetch)}] {coin_id} -> {slug}...", end="", flush=True)

        result = try_all_keys_for_coin(slug, start_date, end_date)

        if result == "ALL_LIMITED":
            print(f"\n\n  [STOP] All {len(API_KEYS)} keys rate-limited!")
            print(f"  Progress saved: {len(social_data)} coins fetched")
            print(f"  Run again when cooldown expires (or add more keys)")
            save_cursor({"completed_coins": list(completed), "social_data": social_data})
            # Still write what we have
            break

        if isinstance(result, dict) and result:
            social_data[coin_id] = result
            completed.add(coin_id)
            sample_val = list(result.values())[0]
            days_got = len(result)
            print(f" OK ({days_got} days, sample={sample_val:.0f})")
        else:
            # No data for this slug (might not exist on Santiment)
            completed.add(coin_id)  # mark as done so we don't retry
            print(f" [NO DATA]")

        # Save cursor after EACH coin
        save_cursor({"completed_coins": list(completed), "social_data": social_data})
        time.sleep(1.5)  # gentle rate limiting between coins

    if not social_data:
        print("\n  No social data fetched. Check API keys.")
        return

    # ── Write to CSV files ──────────────────────────────────────────────────
    print(f"\n  Writing social data to {len(dates)} CSV files...")
    print(f"  Coins with data: {len(social_data)}")
    updated = 0
    skipped = 0

    for date_str in dates:
        csv_path = os.path.join(HIST_DIR, f"{date_str}.csv")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"    [SKIP] {date_str}: {e}")
            skipped += 1
            continue

        sv_col = f"social_volume_{date_str.replace('-', '_')}"

        # Remove ALL old social_volume columns first (clean slate)
        old_sv_cols = [c for c in df.columns if 'social_volume' in c.lower() and c != sv_col]
        if old_sv_cols:
            df = df.drop(columns=old_sv_cols)

        # Map coin_id -> social value for this date
        sv_values = {}
        for coin_id, date_map in social_data.items():
            if date_str in date_map:
                val = date_map[date_str]
                if val >= 0:  # validate
                    sv_values[coin_id] = val

        if sv_values:
            df[sv_col] = df['coin_id'].map(sv_values)
            df[sv_col] = df[sv_col].fillna(0.0)
            df.to_csv(csv_path, index=False)
            updated += 1

    print(f"  Updated: {updated}/{len(dates)} files")
    if skipped:
        print(f"  Skipped: {skipped} files (read errors)")

    # Clean up cursor on success
    if len(to_fetch) == 0 or len(completed) >= len(coins_with_slugs):
        if os.path.exists(CURSOR_FILE):
            os.remove(CURSOR_FILE)
            print(f"\n  Cursor cleared (all coins done)")

    print(f"\n{'=' * 60}")
    print(f"  DONE -- {len(social_data)} coins x {len(dates)} days")
    print(f"  Next: python FillData.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
