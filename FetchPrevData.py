import requests
import csv
import os
import sys
import time
import subprocess
import hashlib
from datetime import datetime, timedelta, timezone
from collections import deque

COINGECKO_API = "https://api.coingecko.com/api/v3"
PROXIES_HOME = "proxies.txt"
MAX_PRX_F = 90
MAX_PRX_TRIES_PER_COIN = 20
BATCH_SIZE = 5
IP_COOLDOWN = 60
PROXY_REFRESH_INTERVAL = 600
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_hourly")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_proxies_from_txt():
    if not os.path.exists(PROXIES_HOME):
        print("[PROXY] proxies.txt not found.")
        return []
    with open(PROXIES_HOME, "r") as f:
        return [line.strip() for line in f if line.strip()]

def refresh_proxies():
    print("\n[PROXY-REFRESH] Running proxy_api.py for fresh proxies...")
    try:
        subprocess.run([sys.executable, "proxy_api.py"], check=True,
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"[PROXY-REFRESH] proxy_api.py failed: {e}")
    new_proxies = load_proxies_from_txt()
    print(f"[PROXY-REFRESH] Reloaded {len(new_proxies)} proxies. Timer reset.\n")
    return new_proxies

def get_proxy_dict(proxy_url):
    return {"http": proxy_url, "https": proxy_url}

def fetch_json_raw(url, params, proxy_url=None, timeout=4):
    """Low-level GET → (json_dict, raw_text). Returns (None, None) on network fail."""
    try:
        kw = dict(params=params, timeout=timeout)
        if proxy_url:
            kw["proxies"] = get_proxy_dict(proxy_url)
        res = requests.get(url, **kw)
        res.raise_for_status()
        raw = res.text
        return res.json(), raw
    except Exception:
        return None, None

def validate_coingecko_response(data, raw_text, date_str, coin_id, proxy_fingerprints, proxy_url):
    """
    Validate that CoinGecko response is genuine.
    Returns (prices, volumes, error_reason) — error_reason is None if OK.
    """
    if data is None:
        return None, None, "network_fail"

    # 1. CoinGecko rate limit / error wrapped in 200
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

    # 3. Validate timestamps fall within requested date (+/- 1 day tolerance)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_start_ms = int(date_obj.timestamp()) * 1000
    day_end_ms = int((date_obj + timedelta(days=2)).timestamp()) * 1000  # tolerance
    day_before_ms = int((date_obj - timedelta(days=1)).timestamp()) * 1000

    first_ts = prices[0][0]
    last_ts = prices[-1][0]

    if first_ts < day_before_ms or last_ts > day_end_ms:
        return None, None, f"wrong_date(got {datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')})"

    # 4. Detect cached proxy responses — same data for different coin
    if proxy_url:
        # Hash the first 5 price values as fingerprint
        fingerprint = hashlib.md5(str(prices[:5]).encode()).hexdigest()
        key = proxy_url

        if key in proxy_fingerprints:
            last_coin, last_fp = proxy_fingerprints[key]
            if last_fp == fingerprint and last_coin != coin_id:
                # Same data for different coin = CACHED
                return None, None, f"cached(same data as {last_coin})"

        proxy_fingerprints[key] = (coin_id, fingerprint)

    return prices, volumes, None

def fetch_coin_list():
    url = f"{COINGECKO_API}/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc",
              "per_page": 50, "page": 1}
    data, _ = fetch_json_raw(url, params)
    if data and isinstance(data, list):
        return data
    for p in load_proxies_from_txt()[:20]:
        data, _ = fetch_json_raw(url, params, p)
        if data and isinstance(data, list):
            return data
    return []

# ── main orchestrator ────────────────────────────────────────────────────────

def save_all_hourly(date_str):
    border = "=" * 60
    print(f"\n{border}")
    print(f"  FETCH HOURLY DATA FOR {date_str}")
    print(f"{border}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")

    proxies = load_proxies_from_txt()
    print(f"[PROXY] Loaded {len(proxies)} proxies")

    print("[COINS] Fetching on MY IP...")
    coin_list = fetch_coin_list()
    if not coin_list:
        print("[COINS] Could not fetch coin list. Aborting.")
        return
    print(f"[COINS] Got {len(coin_list)} coins\n")

    # queues
    main_q      = deque(c["id"] for c in coin_list)
    retry_q     = deque()
    failed_set  = set()
    results     = {}

    prx_i              = 0
    last_ip_time       = 0.0
    ip_count           = 0
    total_prx_fail     = 0
    proxy_refresh_time = time.time()
    proxy_coin_count   = 0
    proxy_fingerprints = {}  # proxy_url → (last_coin, fingerprint)

    def needs_proxy_refresh():
        return (prx_i >= len(proxies)) or \
               (time.time() - proxy_refresh_time >= PROXY_REFRESH_INTERVAL)

    def do_proxy_refresh():
        nonlocal proxies, prx_i, proxy_refresh_time, proxy_fingerprints
        proxies = refresh_proxies()
        prx_i = 0
        proxy_refresh_time = time.time()
        proxy_fingerprints.clear()

    def ip_ready():
        return (time.time() - last_ip_time) >= IP_COOLDOWN or last_ip_time == 0.0

    def try_on_ip(coin_id):
        nonlocal last_ip_time, ip_count
        print(f"    {coin_id} | MY_IP...", end="", flush=True)

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_ts = int(date_obj.timestamp())
        end_ts = int((date_obj + timedelta(days=1)).timestamp())
        url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}

        data, raw = fetch_json_raw(url, params)
        prices, volumes, err = validate_coingecko_response(data, raw, date_str, coin_id, {}, None)

        last_ip_time = time.time()
        ip_count += 1

        if prices:
            sample = prices[0][1]
            print(f" OK ({len(prices)}pts, ${sample:.4f})")
            results[coin_id] = (prices, volumes)
            return True
        else:
            print(f" FAIL ({err})")
            return False

    def try_on_proxy(coin_id):
        nonlocal prx_i, total_prx_fail, proxy_coin_count
        tries = 0
        while tries < MAX_PRX_TRIES_PER_COIN:
            if needs_proxy_refresh():
                do_proxy_refresh()
                proxy_coin_count = 0
            if prx_i >= len(proxies):
                break

            # Rotate proxy after BATCH_SIZE successful fetches
            if proxy_coin_count >= BATCH_SIZE:
                prx_i += 1
                proxy_coin_count = 0
                if prx_i >= len(proxies):
                    do_proxy_refresh()
                print(f"    [ROTATE] Done {BATCH_SIZE} on proxy, next #{prx_i+1}")
                continue

            px = proxies[prx_i]
            print(f"    [TRY] Proxy #{prx_i+1}: {px}...", end="", flush=True)

            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            start_ts = int(date_obj.timestamp())
            end_ts = int((date_obj + timedelta(days=1)).timestamp())
            url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
            params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}

            data, raw = fetch_json_raw(url, params, px)
            prices, volumes, err = validate_coingecko_response(
                data, raw, date_str, coin_id, proxy_fingerprints, px
            )

            if prices:
                sample = prices[0][1]
                print(f" OK ({len(prices)}pts, ${sample:.4f})")
                results[coin_id] = (prices, volumes)
                proxy_coin_count += 1
                return True
            else:
                reason = err or "unknown"
                print(f" FAIL ({reason})")
                if reason.startswith("cached") or reason == "rate_limited":
                    # This proxy is burnt — move on, don't count as tries
                    prx_i += 1
                    proxy_coin_count = 0
                    proxy_fingerprints.pop(px, None)
                else:
                    prx_i += 1
                    proxy_coin_count = 0
                    total_prx_fail += 1
                    tries += 1
        return False

    # ── main loop ────────────────────────────────────────────────────────

    coin_num = 0
    total    = len(main_q)
    mode     = "ip"
    ip_count = 0

    print(f"[START] Mode=MY_IP | {total} coins\n")

    while main_q or retry_q:
        if total_prx_fail >= MAX_PRX_F:
            print(f"\n[ABORT] Total proxy failures ({total_prx_fail}) hit limit {MAX_PRX_F}.")
            break

        if mode == "proxy" and needs_proxy_refresh():
            do_proxy_refresh()
            proxy_coin_count = 0

        if mode == "ip" and ip_count >= BATCH_SIZE:
            mode = "proxy"
            ip_count = 0
            proxy_coin_count = 0
            print(f"[SWITCH] Done {BATCH_SIZE} on MY_IP -> PROXY #{prx_i+1}")

        if mode == "proxy" and ip_ready():
            mode = "ip"
            ip_count = 0
            if retry_q:
                print(f"[SWITCH] IP cooldown OK -> MY_IP  (retry queue: {len(retry_q)} coins)")
            else:
                print(f"[SWITCH] IP cooldown OK -> MY_IP")

        if mode == "ip":
            while retry_q and ip_count < BATCH_SIZE:
                r_coin = retry_q.popleft()
                print(f"[RETRY] {r_coin}", end="")
                ok = try_on_ip(r_coin)
                if not ok:
                    failed_set.add(r_coin)
                    print(f"    [DROP] {r_coin} — failed on both proxy & MY_IP")
                time.sleep(0.3)

            while main_q and ip_count < BATCH_SIZE:
                coin_id = main_q.popleft()
                coin_num += 1
                print(f"[{coin_num}/{total}] {coin_id}", end="")
                ok = try_on_ip(coin_id)
                if not ok:
                    retry_q.append(coin_id)
                    print(f"    [QUEUE] {coin_id} -> retry queue")
                time.sleep(0.3)
            continue

        if mode == "proxy":
            if not main_q and not retry_q:
                break
            if ip_ready():
                continue

            if main_q:
                coin_id = main_q.popleft()
                coin_num += 1
                print(f"[{coin_num}/{total}] {coin_id}")
                ok = try_on_proxy(coin_id)
                if not ok:
                    retry_q.append(coin_id)
                    print(f"    [QUEUE] {coin_id} -> retry queue")
                time.sleep(0.3)
            else:
                wait = IP_COOLDOWN - (time.time() - last_ip_time)
                if wait > 0:
                    print(f"[WAIT] {wait:.0f}s for IP cooldown ({len(retry_q)} in retry queue)...")
                    time.sleep(min(wait, 10))

    # ── write CSV ────────────────────────────────────────────────────────

    ok_count   = len(results)
    fail_count = len(failed_set)

    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "coin_id", "price_usd", "total_volume_usd"])

        for coin in coin_list:
            cid = coin["id"]
            if cid not in results:
                continue
            prices, volumes = results[cid]
            for (ts_p, price), (ts_v, volume) in zip(prices, volumes):
                dt = datetime.fromtimestamp(ts_p / 1000, tz=timezone.utc)
                writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S"), cid, price, volume])

    print(f"\n{border}")
    print(f"  DONE: {ok_count} OK, {fail_count} failed | {file_path}")
    print(f"{border}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FetchPrevData.py <YYYY-MM-DD>")
    else:
        save_all_hourly(sys.argv[1])
