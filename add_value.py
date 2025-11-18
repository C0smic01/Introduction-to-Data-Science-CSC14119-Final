import csv  # file bạn đã viết
import time
import random

from transfermarkt_scraper import get_market_value

INPUT_FILE = "Serie_A.csv"
OUTPUT_FILE = "Serie_A_with_mv.csv"


def random_delay(a=3, b=7):
    time.sleep(random.uniform(a, b))


def fill_market_values():
    rows = []

    # 1. Đọc file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 2. Lặp từng cầu thủ và lấy market value
    for row in rows:
        if row["market_value"] in ("", None):
            name = row["player_name"]
            print(f"Fetching market value for: {name}")

            mv = get_market_value(name)
            row["market_value"] = mv if mv else ""

            random_delay()  # tránh bị block

    # 3. Ghi file mới
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Done! Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    fill_market_values()
