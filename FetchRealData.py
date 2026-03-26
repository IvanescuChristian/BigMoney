"""
FetchRealData.py
────────────────
Fetches REAL market prices (CoinGecko) and REAL social volume (Santiment)
for dates that were predicted, then builds comparison files in real_predictions/.

Columns per day CSV:
  timestamp_utc, coin_id,
  predicted_price, real_price, price_diff, price_diff_pct,
  predicted_social, real_social, social_diff, social_diff_pct,
  pred_std_error
"""

import requests
import os
import sys
import time
import subprocess
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

# ── proxy helpers (same as FetchPrevData) ────────────────────────────────────

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

def fetch_json(url, params, proxy_url=None, timeout=4):
    try:
        kw = dict(params=params, timeout=timeout)
        if proxy_url:
            kw["proxies"] = {"http": proxy_url, "https": proxy_url}
        r = requests.get(url, **kw)
        r.raise_for_status()
        return r.json()
    except:
        return None

def fetch_coin_day(coin_id, date_str, proxy_url=None):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": int(d.timestamp()),
              "to": int((d + timedelta(days=1)).timestamp())}
    data = fetch_json(url, params, proxy_url)
    if data and data.get("prices"):
        return data["prices"], data["total_volumes"]
    return None, None

# ── Santiment social volume ──────────────────────────────────────────────────

_slug_cache = None

def get_slug_map():
    global _slug_cache
    if _slug_cache is not None:
        return _slug_cache
    if not HAS_SAN:
        return {}
    try:
        projects = san.get("projects/all")
        m = {}
        for _, p in projects.iterrows():
            if pd.notna(p.get('name')):
                m[p['name'].lower()] = p['slug']
            if pd.notna(p.get('ticker')):
                m[p['ticker'].lower()] = p['slug']
        _slug_cache = m
        print(f"[SANTIMENT] Got {len(m)} slug entries")
        return m
    except Exception as e:
        print(f"[SANTIMENT] slug fetch error: {e}")
        _slug_cache = {}
        return {}

def fetch_real_social(coin_id, date_str):
    """Fetch real daily social_volume from Santiment. Returns float or None."""
    if not HAS_SAN:
        return None
    slug_map = get_slug_map()
    processed = coin_id.replace("-", " ")
    slug = slug_map.get(processed.lower())
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
    date_coins, pred_data = load_predictions()
    if not date_coins:
        return

    # ── 1) Fetch real prices via CoinGecko (proxy rotation) ──────────
    jobs = []
    for d, coins in sorted(date_coins.items()):
        for c in coins:
            jobs.append((d, c))

    print(f"\n[PRICES] {len(jobs)} coin-day pairs to fetch\n")

    proxies            = load_proxies()
    prx_i              = 0
    proxy_refresh_time = time.time()
    last_ip_time       = 0.0
    ip_count           = 0
    real_prices        = {}   # (date, coin) → (prices, volumes)
    failed             = set()
    retry_q            = deque()
    main_q             = deque(jobs)

    def needs_refresh():
        return prx_i >= len(proxies) or (time.time() - proxy_refresh_time >= PROXY_REFRESH_INTERVAL)

    def do_refresh():
        nonlocal proxies, prx_i, proxy_refresh_time
        proxies = refresh_proxies()
        prx_i = 0
        proxy_refresh_time = time.time()

    def ip_ready():
        return (time.time() - last_ip_time) >= IP_COOLDOWN or last_ip_time == 0.0

    def try_ip(d, c):
        nonlocal last_ip_time, ip_count
        print(f"    {c}({d}) | MY_IP...", end="", flush=True)
        p, v = fetch_coin_day(c, d)
        last_ip_time = time.time()
        ip_count += 1
        if p:
            print(f" OK ({len(p)}pts)")
            real_prices[(d, c)] = (p, v)
            return True
        print(" FAIL")
        return False

    def try_proxy(d, c):
        nonlocal prx_i
        tries = 0
        while tries < MAX_PRX_TRIES_PER_COIN:
            if needs_refresh():
                do_refresh()
            if prx_i >= len(proxies):
                break
            px = proxies[prx_i]
            print(f"    [TRY] #{prx_i+1}: {px}...", end="", flush=True)
            p, v = fetch_coin_day(c, d, px)
            if p:
                print(f" OK ({len(p)}pts)")
                real_prices[(d, c)] = (p, v)
                return True
            print(" FAIL")
            prx_i += 1
            tries += 1
        return False

    total = len(main_q)
    num   = 0
    mode  = "ip"
    ip_count = 0

    while main_q or retry_q:
        if mode == "proxy" and needs_refresh():
            do_refresh()
        if mode == "ip" and ip_count >= BATCH_SIZE:
            mode = "proxy"; ip_count = 0
        if mode == "proxy" and ip_ready():
            mode = "ip"; ip_count = 0

        if mode == "ip":
            while retry_q and ip_count < BATCH_SIZE:
                d, c = retry_q.popleft()
                if not try_ip(d, c):
                    failed.add((d, c))
                time.sleep(0.3)
            while main_q and ip_count < BATCH_SIZE:
                d, c = main_q.popleft()
                num += 1
                print(f"[{num}/{total}]", end="")
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
                time.sleep(0.3)
            else:
                wait = IP_COOLDOWN - (time.time() - last_ip_time)
                if wait > 0:
                    time.sleep(min(wait, 10))

    # ── 2) Fetch real social volume via Santiment ────────────────────
    # one value per (date, coin) — Santiment gives daily aggregates
    real_social = {}   # (date, coin) → float
    if HAS_SAN:
        unique_pairs = set()
        for d, coins in date_coins.items():
            for c in coins:
                unique_pairs.add((d, c))
        print(f"\n[SOCIAL] Fetching real social_volume for {len(unique_pairs)} coin-day pairs...")
        for i, (d, c) in enumerate(sorted(unique_pairs), 1):
            sv = fetch_real_social(c, d)
            if sv is not None:
                real_social[(d, c)] = sv
                print(f"  [{i}] {c}({d}) sv={sv}")
            else:
                print(f"  [{i}] {c}({d}) [SKIP]")
    else:
        print("\n[SOCIAL] Skipped (san not installed)")

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

            # ── real price match ──
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

            # ── real social match ──
            real_social_val = real_social.get((date_str, coin_id), np.nan)

            # ── diffs ──
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
                avg_predicted_social=('predicted_social', 'mean'),
                avg_real_social=('real_social', 'mean'),
                avg_social_error_pct=('social_diff_pct', 'mean'),
                num_hours=('coin_id', 'count')
            ).round(4).reset_index()
            summary.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
            print(f"\n  [SUMMARY] comparison_summary.csv ({len(summary)} coins)")

    print(f"\n{border}")
    print(f"  DONE | Results in: {OUTPUT_DIR}")
    print(f"{border}")


if __name__ == "__main__":
    run()
