import requests
import csv
import os
import time
from datetime import datetime, timedelta

COINGECKO_API = "https://api.coingecko.com/api/v3"
PROXIES_HOME = "proxies.txt"
MAX_PRX_F = 90
MAX_PRX_F_COIN = 30
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_hourly")

def load_proxies_from_txt():
    if not os.path.exists(PROXIES_HOME):
        print(" proxies.txt not found.")
        return []
    with open(PROXIES_HOME, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_proxy_dict(proxy_url):
    return {"http": proxy_url, "https": proxy_url}

def fetch_with_proxy(url, params, proxy_url=None):
    try:
        if proxy_url:
            proxy_dict = get_proxy_dict(proxy_url)
            print(f"Using proxy: {proxy_url}")
            res = requests.get(url,params=params,proxies=proxy_dict,timeout=2)
        else:
            print("Using your IP")
            res = requests.get(url,params=params,timeout=2)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

def fetch_hourly_data(coin_id,date_str,proxies,mode,lg_prx,prx_i,prx_total_f,prx_coin_f):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_pos = int(date_obj.timestamp())
    end_pos = int((date_obj + timedelta(days=1)).timestamp())
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": start_pos, "to": end_pos}
    def try_fetch(proxy_url=None):
        data = fetch_with_proxy(url, params, proxy_url)
        if data:
            prices = data.get("prices",[])
            volumes = data.get("total_volumes",[])
            return prices,volumes
        return None, None
    if mode == "ip":
        prices, volumes = try_fetch()
        if prices is not None:
            return prices, volumes, "ip", None, prx_i, 0, 0 
        else:
            prx_total_f += 1
            prx_coin_f += 1
    if mode == "proxy" and lg_prx:
        prices, volumes = try_fetch(lg_prx)
        if prices is not None:
            return prices, volumes, "proxy", lg_prx, prx_i, 0, 0 
        else:
            prx_total_f += 1
            prx_coin_f += 1
    proxies_list = load_proxies_from_txt()
    if not proxies_list:
        print(" No proxies available.")
        return [], [], None, None, prx_i, prx_total_f, prx_coin_f
    initial_prx_i = prx_i
    tried_all_proxies_in_list = False
    while not tried_all_proxies_in_list and prx_coin_f < MAX_PRX_F_COIN:
        proxy_url = proxies_list[prx_i]
        prices,volumes = try_fetch(proxy_url)
        if prices is not None:
            return prices,volumes,"proxy",proxy_url,(prx_i+1)%len(proxies_list),0,0
        else:
            prx_total_f+=1
            prx_coin_f+=1
            prx_i = (prx_i+1)%len(proxies_list)
            if prx_i == initial_prx_i:
                tried_all_proxies_in_list = True
                print(" Cycled through all proxies in the list for this coin.")
    if mode != "ip" and prx_coin_f < MAX_PRX_F_COIN:
        prices,volumes = try_fetch()
        if prices is not None:
            return prices,volumes,"ip",None,prx_i,0,0 # Reset total fails on success
        else:
            prx_total_f+=1
            prx_coin_f+=1

    return [],[],None,None,prx_i,prx_total_f,prx_coin_f
def fetch_coin_list(proxies):
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50, #or any amount of pages ( maybe i ll make it parsable)
        "page": 1
    }
    data = fetch_with_proxy(url,params)
    if data:
        return data
    for proxy_url in proxies:
        data = fetch_with_proxy(url,params,proxy_url)
        if data:
            return data
    return []
def save_all_hourly(date_str):
    print(f"Fetching hourly prices for {date_str}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
    proxies = load_proxies_from_txt()
    print(f"Loaded {len(proxies)} proxies from proxies.txt")
    coin_list = fetch_coin_list(proxies)
    if not coin_list:
        print("Could nt fetch coin list with any proxy or IP.")
        return
    mode = "ip"
    lg_prx = None
    last_ip_check_time = time.time()
    prx_i = 0
    prx_total_f = 0
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc","coin_id","price_usd","total_volume_usd"])
        for coin in coin_list:
            coin_id = coin["id"]
            print(f"\nFetching hourly data for {coin_id}...")
            prx_coin_f_count = 0
            while prx_coin_f_count < MAX_PRX_F_COIN:
                if prx_total_f >= MAX_PRX_F:
                    print("Maximum total proxy fails reached (30). Exiting. Byeeee!")
                    return
                current_time = time.time()
                if mode == "proxy" and (current_time-last_ip_check_time)>20:
                    print("Retesting IP")
                    ip_test = fetch_with_proxy(f"{COINGECKO_API}/ping",{})
                    if ip_test:
                        print("IP is back online! Switching to IP")
                        mode = "ip"
                        lg_prx = None
                    last_ip_check_time = current_time
                prices,volumes,new_mode,new_proxy,prx_i,prx_total_f_returned,prx_coin_f_count_returned = fetch_hourly_data(coin_id, date_str, proxies, mode, lg_prx, prx_i,prx_total_f, prx_coin_f_count)
                prx_total_f = prx_total_f_returned
                prx_coin_f_count = prx_coin_f_count_returned
                if prices:
                    mode = new_mode
                    lg_prx = new_proxy
                    for (ts_p,price), (ts_v,volume) in zip(prices,volumes):
                        if ts_p != ts_v:
                            print(f"Timestamp mismatch for coin {coin_id} data points!")
                        dt = datetime.utcfromtimestamp(ts_p / 1000)#ms in s
                        writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S"),coin_id,price,volume])
                    print(f"{coin_id}-{len(prices)} points saved")
                    time.sleep(1)
                    break  #Move to the next coin
                else:
                    print(f"No data for {coin_id} on this try. Retrying")
                    time.sleep(1)
            else:
                print(f"Failed {MAX_PRX_F_COIN} times on coin {coin_id}. Skipping to next coin.")
    print(f"\nData saved to: {file_path}")
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("python historical_hourly_fetch.py <YYYY-MM-DD>")
    else:
        date_arg = sys.argv[1]
        save_all_hourly(date_arg)