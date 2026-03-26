import pandas as pd
import san
import sys
from datetime import datetime, timedelta
import time
import os
san.ApiConfig.api_key = 'yvib2jijbrcivyfh_uw25dl5mk4qch5mf'

# ── HARDCODED: CoinGecko ID → Santiment slug ────────────────────────────────
# The dynamic lookup is broken — memecoins overwrite real coins.
# This map is the source of truth for top 50.

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
    "the-open-network":  "the-open-network",
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
    "pi-network":        "pinetwork",
    "internet-computer": "internet-computer",
    "crypto-com-chain":  "crypto-com-coin",
    "paypal-usd":        "paypal-usd",
    "kaspa":             "kaspa",
    "pepe":              "pepe",
    "aptos":             "aptos",
    "dogwifcoin":        "dogwifhat",
    "whitebit":          "whitebit-coin",
    "ondo-finance":      "ondo-finance",
}

def get_dynamic_slug_map():
    """Fallback: build slug map from Santiment API for coins NOT in hardcoded list."""
    try:
        projects = san.get("projects/all")
        slug_map = {}
        for _, p in projects.iterrows():
            slug = p.get('slug', '')
            if not slug:
                continue
            # Map by slug itself (safest)
            slug_map[slug.lower()] = slug
            # Map by ticker (but DON'T overwrite existing — first wins)
            ticker = p.get('ticker', '')
            if pd.notna(ticker) and ticker.lower() not in slug_map:
                slug_map[ticker.lower()] = slug
        print(f"  [SANTIMENT] Dynamic map: {len(slug_map)} entries")
        return slug_map
    except Exception as e:
        print(f"  [SANTIMENT] Dynamic map failed: {e}")
        return {}


def resolve_slug(coin_id, dynamic_map):
    """Resolve CoinGecko coin_id to Santiment slug. Hardcoded first, then dynamic."""
    # 1. Hardcoded (trusted)
    if coin_id in COINGECKO_TO_SANTIMENT:
        return COINGECKO_TO_SANTIMENT[coin_id]

    # 2. Try direct match (coin_id often matches slug)
    processed = coin_id.replace("-", " ")
    if coin_id in dynamic_map:
        return dynamic_map[coin_id]

    # 3. Try processed name
    if processed.lower() in dynamic_map:
        return dynamic_map[processed.lower()]

    return None


def get_social_volume(coin_id, date_str, slug):
    """Fetch daily social volume from Santiment."""
    s_date = f"{date_str}T00:00:00Z"
    e_date = f"{date_str}T23:59:59Z"
    try:
        data = san.get(f"social_volume_total/{slug}",
                       s_date=s_date, e_date=e_date, interval="1d")
        if not data.empty and 'value' in data.columns:
            sv = float(data['value'].iloc[0])
            return sv
        return None
    except Exception as e:
        print(f"    [ERROR] {coin_id} ({slug}): {e}")
        return None
    finally:
        time.sleep(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python FetchSpecialData.py <YYYY-MM-DD>")
        sys.exit(1)

    target_date = sys.argv[1]
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "historical_hourly", f"{target_date}.csv")

    print(f"Loaded: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    if 'coin_id' not in df.columns:
        print("Error: no coin_id column")
        sys.exit(1)

    coins = df['coin_id'].unique()
    print(f"Found {len(coins)} coins\n")

    print("  Building Santiment slug map...")
    dynamic_map = get_dynamic_slug_map()

    sv_col = f"social_volume_{target_date.replace('-', '_')}"
    sv_data = {}

    for i, coin in enumerate(coins, 1):
        slug = resolve_slug(coin, dynamic_map)
        if not slug:
            print(f"[{i}/{len(coins)}] {coin} -> [SKIP] no slug")
            continue

        sv = get_social_volume(coin, target_date, slug)
        if sv is not None:
            sv_data[coin] = sv
            print(f"[{i}/{len(coins)}] {coin} -> {slug} sv={sv}")
        else:
            print(f"[{i}/{len(coins)}] {coin} -> {slug} [NO DATA]")

    # Write to CSV
    df[sv_col] = df['coin_id'].map(sv_data)
    df.to_csv(csv_path, index=False)

    found = sum(1 for v in sv_data.values() if v is not None)
    print(f"\nDone: {found} found, {len(coins) - found} skipped/no data")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
