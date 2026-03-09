from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def fetch_proxies():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")  #maximize for ss
    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://free-proxy-list.net/")
        print("Page loadded")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME,"table-responsive"))#check html
        )
        print("Div with class 'table-responsive' found")
        table_div = driver.find_element(By.CLASS_NAME,"table-responsive")
        table = table_div.find_element(By.TAG_NAME,"table")
        tbody = table.find_element(By.TAG_NAME,"tbody")
        rows = tbody.find_elements(By.TAG_NAME,"tr")
        print(f"Found {len(rows)} rows in the proxy table")
        proxies = []
        for i, row in enumerate(rows):
            cells = row.find_elements(By.TAG_NAME,"td")
            if len(cells) >= 2:
                ip = cells[0].text.strip()
                port = cells[1].text.strip()
                proxies.append(f"{ip}:{port}")
                print(f"Proxy #{i+1}:{ip}:{port}")
            else:
                print(f"Row #{i+1} empty")
        with open("proxies.txt", "w") as f:
            for proxy in proxies:
                f.write(proxy+"\n")
        print(f"Saved {len(proxies)} proxies to proxies.txt")

    except Exception as e:
        print("Error:",repr(e))#return the string of order
        driver.save_screenshot("error_screenshot.png")
        with open("page_source.html","w",encoding="utf-8") as f:
            f.write(driver.page_source)#saved page loc
    finally:
        driver.quit()

if __name__ == "__main__":
    fetch_proxies()
