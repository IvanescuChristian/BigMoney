"""
FetchSpecialData.py — Extrage Social Volume + Sentiment de la Santiment API.
GraphQL direct (nu depinde de sanpy). NU blocheaza — max 2 retry per coin apoi skip.
Nu foloseste proxy (Santiment nu blocheaza per IP la ritmul nostru).
"""
import pandas as pd
import numpy as np
import requests
import sys
import os
import time

SANTIMENT_GQL = "https://api.santiment.net/graphql"
SANTIMENT_API_KEY = ""  # Lasa gol pt free tier sau pune key-ul tau
SLEEP_BETWEEN = 1.5     # Secunde intre requesturi
MAX_RETRIES = 2          # Max retry per coin (nu 10, nu infinit)
REQUEST_TIMEOUT = 10     # Secunde timeout per request

# Manual mapping pt monede problematice
MANUAL_SLUG_MAP = {
    'binance-bridged-usdt-bnb-smart-chain': 'tether',
    'coinbase-wrapped-btc': 'bitcoin',
    'wrapped-bitcoin': 'bitcoin',
    'wrapped-eeth': 'ethereum',
    'wrapped-steth': 'staked-ether',
    'staked-ether': 'staked-ether',
    'weth': 'ethereum',
    'jito-staked-sol': 'solana',
    'blackrock-usd-institutional-digital-liquidity-fund': None,
    'ethena-usde': None,
    'usds': None,
    'figure-heloc': None,  # Nu exista pe Santiment
}

def gql_request(query):
    """Trimite GraphQL request la Santiment. Returneaza dict sau None."""
    headers = {"Content-Type": "application/json"}
    if SANTIMENT_API_KEY:
        headers["Authorization"] = f"Apikey {SANTIMENT_API_KEY}"
    try:
        resp = requests.post(SANTIMENT_GQL, json={"query": query}, headers=headers, timeout=REQUEST_TIMEOUT)
        data = resp.json()
        if 'errors' in data:
            err = str(data['errors'])[:100]
            if 'rate limit' in err.lower() or '429' in err:
                return "RATE_LIMITED"
            return None
        return data
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None

def build_slug_map():
    """Construieste CoinGecko ID -> Santiment slug mapping."""
    print("  Building Santiment slug map...")
    query = '{ allProjects(minVolume: 0) { slug name ticker } }'
    data = gql_request(query)
    if data is None or data == "RATE_LIMITED":
        print("  [WARN] Could not fetch slug map")
        return {}
    
    projects = data.get("data", {}).get("allProjects", [])
    slug_map = {}
    for p in projects:
        slug = p.get("slug", "")
        name = (p.get("name") or "").lower()
        ticker = (p.get("ticker") or "").lower()
        if name: slug_map[name] = slug
        if ticker: slug_map[ticker] = slug
        if slug: slug_map[slug] = slug
        slug_map[name.replace(" ", "-")] = slug
    
    print(f"  Got {len(slug_map)} slug entries from {len(projects)} projects")
    return slug_map

def resolve_slug(coin_id, slug_map):
    """Rezolva CoinGecko coin_id -> Santiment slug."""
    if coin_id in MANUAL_SLUG_MAP:
        return MANUAL_SLUG_MAP[coin_id]
    if coin_id in slug_map:
        return slug_map[coin_id]
    cleaned = coin_id.replace("-", " ")
    if cleaned in slug_map:
        return slug_map[cleaned]
    cleaned2 = coin_id.replace("-", "_")
    if cleaned2 in slug_map:
        return slug_map[cleaned2]
    first = coin_id.split("-")[0]
    if first in slug_map and first not in ("bitcoin", "ethereum", "wrapped"):
        return slug_map[first]
    return None

def fetch_metric(slug, date_str, metric_name):
    """Fetch un metric de la Santiment. Max MAX_RETRIES incercari."""
    query = """{
        getMetric(metric: "%s") {
            timeseriesData(slug: "%s", from: "%sT00:00:00Z", to: "%sT23:59:59Z", interval: "1d") {
                datetime
                value
            }
        }
    }""" % (metric_name, slug, date_str, date_str)
    
    for attempt in range(MAX_RETRIES):
        result = gql_request(query)
        if result == "RATE_LIMITED":
            wait = (attempt + 1) * 5
            print(f" rate-limited, wait {wait}s", end="")
            time.sleep(wait)
            continue
        if result is None:
            continue
        ts_data = result.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
        if ts_data:
            return ts_data[0].get("value", None)
        return None
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python FetchSpecialData.py <YYYY-MM-DD>")
        sys.exit(1)
    
    target_date = sys.argv[1]
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_hourly", f"{target_date}.csv")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
    
    if 'coin_id' not in df.columns:
        print("Error: CSV must have 'coin_id' column.")
        sys.exit(1)
    
    distinct_coins = df['coin_id'].unique()
    print(f"Found {len(distinct_coins)} coins\n")
    
    slug_map = build_slug_map()
    
    sv_data = {}
    sent_data = {}
    found = 0; skipped = 0; no_data = 0
    
    for i, coin in enumerate(distinct_coins):
        slug = resolve_slug(coin, slug_map)
        
        tag = f"[{i+1}/{len(distinct_coins)}]"
        
        if slug is None:
            print(f"{tag} {coin} -> [SKIP] no slug")
            sv_data[coin] = None
            sent_data[coin] = None
            skipped += 1
            continue
        
        print(f"{tag} {coin} -> {slug}", end="")
        
        # Social Volume
        sv = fetch_metric(slug, target_date, "social_volume_total")
        sv_data[coin] = sv
        
        # Sentiment Balance (bonus, gratis)
        sb = fetch_metric(slug, target_date, "sentiment_balance_total")
        sent_data[coin] = sb
        
        if sv is not None:
            print(f" sv={sv}", end="")
            found += 1
        else:
            print(f" sv=None", end="")
            no_data += 1
        
        if sb is not None:
            print(f" sent={sb:.1f}")
        else:
            print(f" sent=None")
        
        time.sleep(SLEEP_BETWEEN)
    
    print(f"\nDone: {found} found, {skipped} skipped, {no_data} no data")
    
    # Adauga coloanele
    sv_col = f"social_volume_{target_date.replace('-','_')}"
    sent_col = f"sentiment_{target_date.replace('-','_')}"
    
    df[sv_col] = df['coin_id'].map(sv_data)
    df[sent_col] = df['coin_id'].map(sent_data)
    
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
