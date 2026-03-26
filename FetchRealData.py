"""
FetchRealData.py
────────────────
Fetches REAL market prices (CoinGecko) and REAL social volume (Santiment)
for dates that were predicted, then builds comparison files in real_predictions/.
"""

import requests
import os
import sys
import time
import subprocess
import hashlib
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta, timezone
from collections import deque

try:
    import san
    san.ApiConfig.api_key = 'yvib2jijbrcivyfh_uw25dl5mk4qch5mf'
    HAS_SAN = True
except ImportError:
    HAS_SAN = False
    print("[WARN] san module not installed — social volume comparison disabled")

COINGECKO_API = "https://api.coingecko.com/api/v3"
PROXIES_HOME = "proxies.txt"
MAX_PRX_TRIES_PER_COIN = 20
BATCH_SIZE = 5
IP_COOLDOWN = 60
PROXY_REFRESH_INTERVAL = 600

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PRED_DIR   = os.path.join(BASE_DIR, "predicted_hourly")
OUTPUT_DIR = os.path.join(BASE_DIR, "real_predictions")

# ── proxy helpers ────────────────────────────────────────────────────────────

def load_proxies():
    if not os.path.exists(PROXIES_HOME):
        return []
    with open(PROXIES_HOME, "r") as f:
        return [l.strip() for l in f if l.strip()]

def refresh_proxies():
    print("[PROXY-REFRESH] Running proxy_api.py...")
    try:
        subprocess.run([sys.executable, "proxy_api.py"], check=True,
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"[PROXY-REFRESH] failed: {e}")
    new = load_proxies()
    print(f"[PROXY-REFRESH] Reloaded {len(new)} proxies.\n")
    return new

def fetch_json_raw(url, params, proxy_url=None, timeout=4):
    try:
        kw = dict(params=params, timeout=timeout)
        if proxy_url:
            kw["proxies"] = {"http": proxy_url, "https": proxy_url}
        r = requests.get(url, **kw)
        r.raise_for_status()
        return r.json(), r.text
    except:
        return None, None

def validate_coingecko_response(data, raw_text, date_str, coin_id, proxy_fingerprints, proxy_url):
    """
    Validate CoinGecko response is genuine, not cached/rate-limited/wrong-date.
    Returns (prices, volumes, error_reason).
    """
    if data is None:
        return None, None, "network_fail"

    # 1. CoinGecko error wrapped in 200
    if isinstance(data, dict) and "status" in data:
        status = data["status"]
        if isinstance(status, dict):
            code = status.get("error_code", 0)
            msg = status.get("error_message", "")
            if code == 429 or "rate" in str(msg).lower():
                return None, None, "rate_limited"
            if code >= 400:
                return None, None, f"api_error_{code}"

    # 2. Must have prices
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    if not prices:
        return None, None, "no_prices"

    # 3. Timestamps must match requested date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_start_ms = int((date_obj - timedelta(days=1)).timestamp()) * 1000
    day_end_ms = int((date_obj + timedelta(days=2)).timestamp()) * 1000

    first_ts = prices[0][0]
    last_ts = prices[-1][0]
    if first_ts < day_start_ms or last_ts > day_end_ms:
        got_date = datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')
        return None, None, f"wrong_date(got {got_date})"

    # 4. Detect cached proxy — same data for different coin
    if proxy_url:
        fingerprint = hashlib.md5(str(prices[:5]).encode()).hexdigest()
        key = proxy_url
        if key in proxy_fingerprints:
            last_coin, last_fp = proxy_fingerprints[key]
            if last_fp == fingerprint and last_coin != coin_id:
                return None, None, f"cached(same as {last_coin})"
        proxy_fingerprints[key] = (coin_id, fingerprint)

    return prices, volumes, None

def fetch_coin_day(coin_id, date_str, proxy_url=None, proxy_fingerprints=None):
    """Fetch + validate. Returns (prices, volumes, error_reason)."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": int(d.timestamp()),
              "to": int((d + timedelta(days=1)).timestamp())}

    data, raw = fetch_json_raw(url, params, proxy_url)
    fp = proxy_fingerprints if proxy_fingerprints else {}
    return validate_coingecko_response(data, raw, date_str, coin_id, fp, proxy_url)

# ── Santiment (same hardcoded map as FetchSpecialData) ───────────────────────

COINGECKO_TO_SANTIMENT = {
    "bitcoin": "bitcoin", "ethereum": "ethereum", "tether": "tether",
    "ripple": "ripple", "binancecoin": "binance-coin", "usd-coin": "usd-coin",
    "solana": "solana", "tron": "tron", "dogecoin": "dogecoin",
    "cardano": "cardano", "bitcoin-cash": "bitcoin-cash",
    "hyperliquid": "hyperliquid", "leo-token": "unus-sed-leo",
    "chainlink": "chainlink", "monero": "monero", "stellar": "stellar",
    "dai": "multi-collateral-dai", "litecoin": "litecoin",
    "avalanche-2": "avalanche", "hedera-hashgraph": "hedera-hashgraph",
    "sui": "sui", "shiba-inu": "shiba-inu",
    "the-open-network": "the-open-network", "polkadot": "polkadot-new",
    "uniswap": "uniswap", "near": "near-protocol", "aave": "aave",
    "bittensor": "bittensor", "okb": "okb", "zcash": "zcash",
    "ethereum-classic": "ethereum-classic", "mantle": "mantle",
    "pax-gold": "pax-gold", "tether-gold": "tether-gold",
    "pi-network": "pinetwork", "internet-computer": "internet-computer",
    "crypto-com-chain": "crypto-com-coin", "paypal-usd": "paypal-usd",
    "kaspa": "kaspa", "pepe": "pepe", "aptos": "aptos",
    "whitebit": "whitebit-coin", "ondo-finance": "ondo-finance",
}

def fetch_real_social(coin_id, date_str):
    if not HAS_SAN:
        return None
    # Use hardcoded slug first
    slug = COINGECKO_TO_SANTIMENT.get(coin_id)
    if not slug:
        return None
    try:
        data = san.get(f"social_volume_total/{slug}",
                       s_date=f"{date_str}T00:00:00Z",
                       e_date=f"{date_str}T23:59:59Z",
                       interval="1d")
        if not data.empty and 'value' in data.columns:
            return float(data['value'].iloc[0])
    except:
        pass
    finally:
        time.sleep(0.5)
    return None

# ── discover predictions ─────────────────────────────────────────────────────

def load_predictions():
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")))
    pred_files = [f for f in pred_files if "summary" not in os.path.basename(f).lower()]
    if not pred_files:
        print("[ERROR] No prediction files in predicted_hourly/")
        return {}, {}
    date_coins, pred_data = {}, {}
    for fp in pred_files:
        bn = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_csv(fp)
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            date_coins[bn] = df['coin_id'].unique().tolist()
            pred_data[bn] = df
            print(f"  [PRED] {bn}.csv — {len(date_coins[bn])} coins")
        except Exception as e:
            print(f"  [SKIP] {fp}: {e}")
    return date_coins, pred_data

# ── main ─────────────────────────────────────────────────────────────────────

def run():
    border = "=" * 60
    print(f"\n{border}")
    print("  FETCH REAL PRICES + SOCIAL & COMPARE WITH PREDICTIONS")
    print(f"{border}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fresh proxies
    print("[PROXY] Refreshing proxies before fetch...")
    refresh_proxies()

    date_coins, pred_data = load_predictions()
    if not date_coins:
        return

    # ── 1) Fetch real prices via CoinGecko ───────────────────────────
    jobs = []
    for d, coins in sorted(date_coins.items()):
        for c in coins:
            jobs.append((d, c))

    print(f"\n[PRICES] {len(jobs)} coin-day pairs to fetch\n")

    proxies            = load_proxies()
    prx_i              = 0
    proxy_refresh_time = time.time()
    proxy_coin_count   = 0
    proxy_fingerprints = {}
    last_ip_time       = 0.0
    ip_count           = 0
    real_prices        = {}
    failed             = set()
    retry_q            = deque()
    main_q             = deque(jobs)

    def needs_refresh():
        return prx_i >= len(proxies) or (time.time() - proxy_refresh_time >= PROXY_REFRESH_INTERVAL)

    def do_refresh():
        nonlocal proxies, prx_i, proxy_refresh_time, proxy_fingerprints
        proxies = refresh_proxies()
        prx_i = 0
        proxy_refresh_time = time.time()
        proxy_fingerprints.clear()

    def ip_ready():
        return (time.time() - last_ip_time) >= IP_COOLDOWN or last_ip_time == 0.0

    def try_ip(d, c):
        nonlocal last_ip_time, ip_count
        print(f"    {c}({d}) | MY_IP...", end="", flush=True)
        prices, volumes, err = fetch_coin_day(c, d, proxy_fingerprints={})
        last_ip_time = time.time()
        ip_count += 1
        if prices:
            sample = prices[0][1]
            print(f" OK ({len(prices)}pts, ${sample:.4f})")
            real_prices[(d, c)] = (prices, volumes)
            return True
        print(f" FAIL ({err})")
        return False

    def try_proxy(d, c):
        nonlocal prx_i, proxy_coin_count
        tries = 0
        no_prices_count = 0
        while tries < MAX_PRX_TRIES_PER_COIN:
            if needs_refresh():
                do_refresh()
                proxy_coin_count = 0
            if prx_i >= len(proxies):
                break

            # Rotate after BATCH_SIZE successful fetches
            if proxy_coin_count >= BATCH_SIZE:
                prx_i += 1
                proxy_coin_count = 0
                proxy_fingerprints.pop(proxies[prx_i-1] if prx_i-1 < len(proxies) else "", None)
                if prx_i >= len(proxies):
                    do_refresh()
                print(f"    [ROTATE] Done {BATCH_SIZE} on proxy, next #{prx_i+1}")
                continue

            px = proxies[prx_i]
            print(f"    [TRY] #{prx_i+1}: {px}...", end="", flush=True)
            prices, volumes, err = fetch_coin_day(c, d, px, proxy_fingerprints)

            if prices:
                sample = prices[0][1]
                print(f" OK ({len(prices)}pts, ${sample:.4f})")
                real_prices[(d, c)] = (prices, volumes)
                proxy_coin_count += 1
                return True
            else:
                reason = err or "unknown"
                print(f" FAIL ({reason})")

                if reason == "no_prices":
                    no_prices_count += 1
                    if no_prices_count >= 2:
                        # 2x no_prices = coin problem, keep proxy, queue coin
                        print(f"    [NO_DATA] {c}({d}) — 2x no_prices, coin problem -> queue")
                        return False
                    # First: maybe proxy issue, try next
                    prx_i += 1
                    proxy_coin_count = 0
                    tries += 1
                elif reason.startswith("cached") or reason == "rate_limited":
                    prx_i += 1
                    proxy_coin_count = 0
                    proxy_fingerprints.pop(px, None)
                else:
                    prx_i += 1
                    proxy_coin_count = 0
                    tries += 1
        return False

    # ── fetch loop ───────────────────────────────────────────────────
    total = len(main_q)
    num   = 0
    mode  = "ip"
    ip_count = 0

    while main_q or retry_q:
        if mode == "proxy" and needs_refresh():
            do_refresh()
            proxy_coin_count = 0

        if mode == "ip" and ip_count >= BATCH_SIZE:
            mode = "proxy"; ip_count = 0; proxy_coin_count = 0
            print(f"[SWITCH] Done {BATCH_SIZE} on MY_IP -> PROXY #{prx_i+1}")

        if mode == "proxy" and ip_ready():
            mode = "ip"; ip_count = 0
            if retry_q:
                print(f"[SWITCH] IP cooldown OK -> MY_IP  (retry queue: {len(retry_q)})")
            else:
                print(f"[SWITCH] IP cooldown OK -> MY_IP")

        if mode == "ip":
            while retry_q and ip_count < BATCH_SIZE:
                d, c = retry_q.popleft()
                print(f"[RETRY] {c}({d})", end="")
                if not try_ip(d, c):
                    failed.add((d, c))
                    print(f"    [DROP] {c}({d})")
                time.sleep(0.3)
            while main_q and ip_count < BATCH_SIZE:
                d, c = main_q.popleft()
                num += 1
                print(f"[{num}/{total}] {c}({d})", end="")
                if not try_ip(d, c):
                    retry_q.append((d, c))
                time.sleep(0.3)
            continue

        if mode == "proxy":
            if not main_q and not retry_q:
                break
            if ip_ready():
                continue
            if main_q:
                d, c = main_q.popleft()
                num += 1
                print(f"[{num}/{total}] {c}({d})")
                if not try_proxy(d, c):
                    retry_q.append((d, c))
                    print(f"    [QUEUE] {c}({d}) -> retry queue")
                time.sleep(0.3)
            else:
                wait = IP_COOLDOWN - (time.time() - last_ip_time)
                if wait > 0:
                    time.sleep(min(wait, 10))

    print(f"\n[PRICES DONE] {len(real_prices)} OK, {len(failed)} failed\n")

    # ── 2) Fetch real social volume ──────────────────────────────────
    real_social = {}
    if HAS_SAN:
        unique_pairs = set()
        for d, coins in date_coins.items():
            for c in coins:
                unique_pairs.add((d, c))
        print(f"[SOCIAL] Fetching real social_volume for {len(unique_pairs)} coin-day pairs...")
        for i, (d, c) in enumerate(sorted(unique_pairs), 1):
            sv = fetch_real_social(c, d)
            if sv is not None:
                real_social[(d, c)] = sv
                print(f"  [{i}] {c}({d}) sv={sv}")
            else:
                print(f"  [{i}] {c}({d}) [SKIP]")

    # ── 3) Build comparison CSVs ─────────────────────────────────────
    print(f"\n[COMPARE] Building comparison files...")

    for date_str, pred_df in sorted(pred_data.items()):
        rows = []
        for _, prow in pred_df.iterrows():
            coin_id     = prow['coin_id']
            ts          = prow['timestamp_utc']
            pred_price  = prow['price_usd']
            pred_error  = prow.get('error', np.nan)
            pred_social = prow.get('predicted_social_volume', np.nan)

            # Real price match
            real_price_val = np.nan
            key = (date_str, coin_id)
            if key in real_prices:
                rp_list, _ = real_prices[key]
                if rp_list:
                    rdf = pd.DataFrame(rp_list, columns=['ts_ms', 'price'])
                    rdf['ts'] = pd.to_datetime(rdf['ts_ms'] / 1000, unit='s', utc=True)
                    ts_aware = pd.Timestamp(ts)
                    if ts_aware.tzinfo is None:
                        ts_aware = ts_aware.tz_localize('UTC')
                    idx = (rdf['ts'] - ts_aware).abs().idxmin()
                    real_price_val = rdf.loc[idx, 'price']

            # Real social match
            real_social_val = real_social.get((date_str, coin_id), np.nan)

            # Diffs
            p_diff = (real_price_val - pred_price) if not np.isnan(real_price_val) else np.nan
            p_diff_pct = (p_diff / pred_price * 100) if (not np.isnan(p_diff) and pred_price != 0) else np.nan

            s_diff = np.nan
            s_diff_pct = np.nan
            if not np.isnan(real_social_val) and not np.isnan(pred_social):
                s_diff = real_social_val - pred_social
                s_diff_pct = (s_diff / pred_social * 100) if pred_social != 0 else np.nan

            rows.append({
                'timestamp_utc': ts,
                'coin_id': coin_id,
                'predicted_price': round(pred_price, 6),
                'real_price': round(real_price_val, 6) if not np.isnan(real_price_val) else np.nan,
                'price_diff': round(p_diff, 6) if not np.isnan(p_diff) else np.nan,
                'price_diff_pct': round(p_diff_pct, 4) if not np.isnan(p_diff_pct) else np.nan,
                'predicted_social': round(pred_social, 2) if not np.isnan(pred_social) else np.nan,
                'real_social': round(real_social_val, 2) if not np.isnan(real_social_val) else np.nan,
                'social_diff': round(s_diff, 2) if not np.isnan(s_diff) else np.nan,
                'social_diff_pct': round(s_diff_pct, 2) if not np.isnan(s_diff_pct) else np.nan,
                'pred_std_error': round(pred_error, 6) if not np.isnan(pred_error) else np.nan
            })

        if rows:
            out_df = pd.DataFrame(rows)
            out_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
            out_df.to_csv(out_path, index=False)
            ok_p = out_df['real_price'].notna().sum()
            ok_s = out_df['real_social'].notna().sum()
            print(f"  [SAVED] {date_str}.csv — price:{ok_p}/{len(out_df)}  social:{ok_s}/{len(out_df)}")

    # ── 4) Summary ───────────────────────────────────────────────────
    all_csvs = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
                if "summary" not in os.path.basename(f).lower()]
    if all_csvs:
        combined = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        valid = combined.dropna(subset=['real_price', 'predicted_price'])
        if not valid.empty:
            summary = valid.groupby('coin_id').agg(
                avg_predicted_price=('predicted_price', 'mean'),
                avg_real_price=('real_price', 'mean'),
                avg_price_error_pct=('price_diff_pct', 'mean'),
                max_price_error_pct=('price_diff_pct', lambda x: x.abs().max()),
                num_hours=('coin_id', 'count')
            ).round(4).reset_index()
            summary.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
            print(f"\n  [SUMMARY] comparison_summary.csv ({len(summary)} coins)")

    print(f"\n{border}")
    print(f"  DONE | Results in: {OUTPUT_DIR}")
    print(f"{border}")


if __name__ == "__main__":
    run()
