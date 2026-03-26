import requests
import csv
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta, timezone
from collections import deque

COINGECKO_API = "https://api.coingecko.com/api/v3"
PROXIES_HOME = "proxies.txt"
MAX_PRX_F = 90
MAX_PRX_TRIES_PER_COIN = 20   # try max 20 proxies per coin before deferring
BATCH_SIZE = 5                 # how many coins per IP/proxy batch
IP_COOLDOWN = 60               # seconds between MY_IP batches
PROXY_REFRESH_INTERVAL = 600   # 10 minute — refresh proxies
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_hourly")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_proxies_from_txt():
    if not os.path.exists(PROXIES_HOME):
        print("[PROXY] proxies.txt not found.")
        return []
    with open(PROXIES_HOME, "r") as f:
        return [line.strip() for line in f if line.strip()]

def refresh_proxies():
    """Run proxy_api.py to get fresh proxies, reload list."""
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

def fetch_json(url, params, proxy_url=None, timeout=4):
    """Low-level GET → JSON.  Returns dict on success, None on failure."""
    try:
        kw = dict(params=params, timeout=timeout)
        if proxy_url:
            kw["proxies"] = get_proxy_dict(proxy_url)
        res = requests.get(url, **kw)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

def fetch_coin_data(coin_id, date_str, proxy_url=None):
    """Fetch hourly price+volume for one coin.  Returns (prices, volumes) or (None, None)."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_ts = int(date_obj.timestamp())
    end_ts   = int((date_obj + timedelta(days=1)).timestamp())
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}

    data = fetch_json(url, params, proxy_url)
    if data and data.get("prices"):
        return data["prices"], data["total_volumes"]
    return None, None

def fetch_coin_list():
    url = f"{COINGECKO_API}/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc",
              "per_page": 50, "page": 1}
    data = fetch_json(url, params)
    if data:
        return data
    for p in load_proxies_from_txt()[:20]:
        data = fetch_json(url, params, p)
        if data:
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
    proxy_refresh_time = time.time()   # timer for 10-min refresh

    def needs_proxy_refresh():
        """Refresh when: all proxies exhausted OR 10 min passed."""
        return (prx_i >= len(proxies)) or \
               (time.time() - proxy_refresh_time >= PROXY_REFRESH_INTERVAL)

    def do_proxy_refresh():
        nonlocal proxies, prx_i, proxy_refresh_time
        proxies = refresh_proxies()
        prx_i = 0
        proxy_refresh_time = time.time()

    def ip_ready():
        return (time.time() - last_ip_time) >= IP_COOLDOWN or last_ip_time == 0.0

    def try_on_ip(coin_id):
        nonlocal last_ip_time, ip_count
        print(f"    {coin_id} | MY_IP...", end="", flush=True)
        p, v = fetch_coin_data(coin_id, date_str)
        if p:
            print(f" OK ({len(p)}pts)")
            results[coin_id] = (p, v)
            last_ip_time = time.time()
            ip_count += 1
            return True
        else:
            print(" FAIL")
            last_ip_time = time.time()
            ip_count += 1
            return False

    def try_on_proxy(coin_id):
        nonlocal prx_i, total_prx_fail
        tries = 0
        while tries < MAX_PRX_TRIES_PER_COIN:
            # refresh daca am epuizat lista sau au trecut 10 min
            if needs_proxy_refresh():
                do_proxy_refresh()
            if prx_i >= len(proxies):
                break   # refresh-ul n-a adus nimic

            px = proxies[prx_i]
            print(f"    [TRY] Proxy #{prx_i+1}: {px}...", end="", flush=True)
            p, v = fetch_coin_data(coin_id, date_str, px)
            if p:
                print(f" OK ({len(p)}pts)")
                results[coin_id] = (p, v)
                # NU avansam prx_i — refolosim proxy-ul bun la urm coin
                return True
            else:
                print(" FAIL")
                prx_i += 1
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

        # ── proxy refresh check (in proxy mode) ─────────────────────
        if mode == "proxy" and needs_proxy_refresh():
            do_proxy_refresh()

        # ── decide mode ──────────────────────────────────────────────
        if mode == "ip" and ip_count >= BATCH_SIZE:
            mode = "proxy"
            ip_count = 0
            print(f"[SWITCH] Done {BATCH_SIZE} on MY_IP -> PROXY #{prx_i+1}")

        if mode == "proxy" and ip_ready():
            mode = "ip"
            ip_count = 0
            if retry_q:
                print(f"[SWITCH] IP cooldown OK -> MY_IP  (retry queue: {len(retry_q)} coins)")
            else:
                print(f"[SWITCH] IP cooldown OK -> MY_IP")

        # ── IP mode ──────────────────────────────────────────────────
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

        # ── proxy mode ───────────────────────────────────────────────
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
                    print(f"    [QUEUE] {coin_id} -> retry queue ({MAX_PRX_TRIES_PER_COIN} proxy fails)")
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
