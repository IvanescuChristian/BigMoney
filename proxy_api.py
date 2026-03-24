"""
proxy_api.py — Extrage proxy-uri gratuite din MULTIPLE surse.
Ruleaza MEREU la inceput (symphony.bat) ca sa ai proxy-uri fresh.
Surse fara selenium (rapide):
  1. ProxyScrape API 
  2. Geonode API
Surse cu selenium (fallback daca API-urile nu dau destule):
  3. free-proxy-list.net
  4. Spys.one
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import sys
import time

PROXY_FILE = "proxies.txt"
MIN_PROXIES_NEEDED = 30

def fetch_from_proxyscrape():
    print("[ProxyScrape] Fetching proxies via API...")
    try:
        url = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&timeout=3000"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            proxies = []
            for line in resp.text.strip().split('\n'):
                line = line.strip()
                if line and ':' in line:
                    proxy = line.split('://')[-1] if '://' in line else line
                    proxies.append(proxy)
            print(f"[ProxyScrape] Got {len(proxies)} raw proxies")
            return proxies
    except Exception as e:
        print(f"[ProxyScrape] Error: {e}")
    return []

def fetch_from_geonode():
    print("[Geonode] Fetching via API...")
    try:
        url = "https://proxylist.geonode.com/api/proxy-list?limit=200&page=1&sort_by=lastChecked&sort_type=desc"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            proxies = []
            for p in data.get('data', []):
                ip = p.get('ip', ''); port = p.get('port', '')
                if ip and port: proxies.append(f"{ip}:{port}")
            print(f"[Geonode] Got {len(proxies)} proxies")
            return proxies
    except Exception as e:
        print(f"[Geonode] Error: {e}")
    return []

def fetch_from_freeproxylist(driver):
    print("[FreeProxyList] Loading page...")
    try:
        driver.get("https://free-proxy-list.net/")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table-responsive"))
        )
        table_div = driver.find_element(By.CLASS_NAME, "table-responsive")
        tbody = table_div.find_element(By.TAG_NAME, "table").find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")
        proxies = []
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                ip = cells[0].text.strip(); port = cells[1].text.strip()
                if ip and port: proxies.append(f"{ip}:{port}")
        print(f"[FreeProxyList] Got {len(proxies)} proxies")
        return proxies
    except Exception as e:
        print(f"[FreeProxyList] Error: {e}")
        try:
            driver.save_screenshot("error_screenshot.png")
            with open("page_source.html","w",encoding="utf-8") as f:
                f.write(driver.page_source)
        except: pass
        return []

def fetch_from_spysone(driver):
    print("[Spys.one] Loading page...")
    try:
        driver.get("https://spys.one/en/free-proxy-list/")
        time.sleep(3)
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.spy1x, tr.spy1xx")
        proxies = []
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells:
                text = cells[0].text.strip()
                if ':' in text and text[0].isdigit():
                    proxy = text.split('\n')[0].strip()
                    if proxy.count(':') == 1: proxies.append(proxy)
        print(f"[Spys.one] Got {len(proxies)} proxies")
        return proxies
    except Exception as e:
        print(f"[Spys.one] Error: {e}")
        return []

def fetch_proxies():
    all_proxies = []
    all_proxies.extend(fetch_from_proxyscrape())
    all_proxies.extend(fetch_from_geonode())
    
    if len(all_proxies) < MIN_PROXIES_NEEDED:
        print(f"\n[Main] Doar {len(all_proxies)} din API-uri. Pornesc Selenium...")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless"); options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        driver = webdriver.Chrome(options=options)
        try:
            all_proxies.extend(fetch_from_freeproxylist(driver))
            if len(all_proxies) < MIN_PROXIES_NEEDED:
                all_proxies.extend(fetch_from_spysone(driver))
        finally:
            driver.quit()
    
    all_proxies = list(dict.fromkeys(all_proxies))
    print(f"\n[Main] Total unic: {len(all_proxies)} proxy-uri")
    
    if not all_proxies:
        print("[Main] EROARE: Nu am gasit niciun proxy!")
        return
    
    with open(PROXY_FILE, "w") as f:
        for proxy in all_proxies:
            f.write(proxy + "\n")
    print(f"[Main] Salvat {len(all_proxies)} proxy-uri in {PROXY_FILE}")

if __name__ == "__main__":
    fetch_proxies()
