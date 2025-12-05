from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import logging
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from utils import random_delay

# ==============================
# 1. Thiết lập logging
# ==============================
logging.basicConfig(
    filename="crawl_height.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# ==============================
# 2. Đọc CSV
# ==============================
df = pd.read_csv("file1.csv")
df_missing_height = df[df['height'].isna() | (df['height'] == '')]
logging.info(f"Tổng cầu thủ thiếu height: {len(df_missing_height)}")

# ==============================
# 3. Chrome Options
# ==============================
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# ==============================
# 4. Hàm lấy player_url
# ==============================
def get_player_url(driver, player_name):
    search_name = player_name.replace(" ", "+")
    search_url = f"https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query={search_name}"
    driver.get(search_url)
    time.sleep(random.uniform(2, 4))  # delay ngẫu nhiên

    try:
        td = driver.find_element(By.CSS_SELECTOR, "td.hauptlink a")
        href = td.get_attribute("href")
        logging.info(f"{player_name} -> URL found: {href}")
        return href
    except Exception as e:
        logging.warning(f"{player_name} -> URL not found: {e}")
        return None

# ==============================
# 5. Hàm crawl chiều cao
# ==============================
def get_height_from_tm(player_url: str, player_name: str):
    """Lấy height từ trang profile player trên Transfermarkt"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(player_url, headers=headers, timeout=10)
        random_delay()
        if resp.status_code != 200:
            logging.warning(f"{player_name} -> Failed to load profile page ({resp.status_code})")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        # Tìm div chính chứa thông tin
        details_div = soup.find("div", class_="data-header__details")
        if not details_div:
            logging.warning(f"{player_name} -> Details div not found")
            return None

        # Tìm li có height
        height_cm = None
        for li in details_div.find_all("li"):
            if "Height" in li.get_text():
                span = li.find("span", itemprop="height")
                if span:
                    height_text = span.get_text(strip=True)  # ví dụ "1,80 m"
                    match = re.search(r"([\d,]+)\s*m", height_text)
                    if match:
                        height_m = match.group(1).replace(",", ".")
                        height_cm = int(float(height_m) * 100)
                        logging.info(f"{player_name} -> Height: {height_cm} cm")
                        return height_cm

        logging.warning(f"{player_name} -> Height not found in details div")
        return None

    except Exception as e:
        logging.error(f"{player_name} -> Error fetching height: {e}")
        return None

# ==============================
# 6. Crawl và cập nhật DataFrame
# ==============================
processed_count = 0
for idx, row in df_missing_height.iterrows():
    player_name = row['player_name']
    
    url = get_player_url(driver, player_name)
    if url:
        df.at[idx, 'player_url'] = url
        height = get_height_from_tm(url, player_name)
        if height:
            df.at[idx, 'height'] = height
            logging.info(f"{player_name} -> Height updated, saving CSV...")
            # Lưu ngay sau khi cập nhật
            df.to_csv("merged_players_filled_height1.csv", index=False)
        else:
            logging.warning(f"{player_name} -> height not updated")
    else:
        logging.warning(f"{player_name} -> URL not found, skip height crawl")
    
    processed_count += 1
    logging.info(f"Processed {processed_count}/{len(df_missing_height)}")