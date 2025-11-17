"""
TransferMarkt Scraper - Main scraper for player data
"""

import requests
from bs4 import BeautifulSoup
import logging
import random
import time

from config import TRANSFERMARKT_QUICKSEARCH_URL
from utils import random_delay

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_market_value(player_name: str):
    """Return the market value of a player from Transfermarkt"""
    url = TRANSFERMARKT_QUICKSEARCH_URL.format(player_name=player_name)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    logging.info(f"Fetching market value for {player_name}: {url}")

    resp = requests.get(url, headers=headers)
    random_delay()
    if resp.status_code != 200:
        logging.warning(f"Failed to load page for {player_name}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        logging.warning(f"No search results table found for {player_name}")
        return None

    for row in table.tbody.find_all("tr"):

        # Market value
        mv_tag = row.find("td", class_="rechts hauptlink")
        if mv_tag:
            return mv_tag.get_text(strip=True)

    logging.warning(f"Player {player_name} not found")
    return None
