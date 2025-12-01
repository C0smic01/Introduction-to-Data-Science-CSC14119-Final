"""
TransferMarkt Scraper - Scraper for player data
"""

from utils.imports import *

from utils.config import HEADERS, TRANSFERMARKT_PLAYER_SEACH_URL
from utils.utils import random_delay


def parse_market_value(mv_str: str) -> float | None:
    """
    Convert Transfermarkt market value string to float (million €).

    Examples:
        '€700k' -> 0.7
        '€1m'   -> 1.0
        '€12.5m'-> 12.5

    Args:
        mv_str (str): Market value string from Transfermarkt.

    Returns:
        float | None: Value in million euros, or None if cannot parse.
    """
    if not mv_str:
        return None

    mv_str = mv_str.replace("€", "").strip().lower()

    try:
        if mv_str.endswith("k"):
            return float(mv_str[:-1].replace(".", "").replace(",", ".")) / 1000
        elif mv_str.endswith("m"):
            return float(mv_str[:-1].replace(",", "."))
        else:
            return float(mv_str.replace(".", "").replace(",", ".")) / 1_000_000
    except ValueError:
        return None


def get_market_value(player_name: str) -> float | None:
    """
    Fetch the market value of a player from Transfermarkt as float million €.
    """
    url = TRANSFERMARKT_PLAYER_SEACH_URL.format(player_name=player_name)
    logging.info(f"Fetching market value for {player_name}: {url}")

    resp = requests.get(url, headers=HEADERS)
    random_delay()
    if resp.status_code != 200:
        logging.warning(f"Failed to load page for {player_name}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        logging.warning(f"No search results table found for {player_name}")
        return None

    # Extract market value from the first search result
    for row in table.tbody.find_all("tr"):
        mv_tag = row.find("td", class_="rechts hauptlink")
        if mv_tag:
            market_value_str = mv_tag.get_text(strip=True)
            return parse_market_value(market_value_str)

    logging.warning(f"Player {player_name} not found")
    return None


def get_profile_url(driver, player_name: str) -> str | None:
    """
    Get the Transfermarkt profile URL of a player using a Selenium driver.

    Args:
        driver: Selenium WebDriver instance.
        player_name (str): Full name of the player.

    Returns:
        str | None: Player profile URL, or None if not found.
    """
    search_url = TRANSFERMARKT_PLAYER_SEACH_URL.format(player_name=player_name)
    driver.get(search_url)
    random_delay(2, 4)

    try:
        td = driver.find_element(By.CSS_SELECTOR, "td.hauptlink a")
        href = td.get_attribute("href")
        logging.info(f"{player_name}: URL found: {href}")
        return href
    except Exception as e:
        logging.warning(f"{player_name}: URL not found: {e}")
        return None


def get_height(player_url: str, player_name: str) -> int | None:
    """
    Fetch the height of a player from their Transfermarkt profile.

    Args:
        player_url (str): URL of the player's Transfermarkt profile.
        player_name (str): Full name of the player.

    Returns:
        int | None: Height in centimeters, or None if not found.
    """
    try:
        resp = requests.get(player_url, headers=HEADERS, timeout=10)
        random_delay()
        if resp.status_code != 200:
            logging.warning(
                f"{player_name} -> Failed to load profile page ({resp.status_code})"
            )
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        details_div = soup.find("div", class_="data-header__details")
        if not details_div:
            logging.warning(f"{player_name} -> Details div not found")
            return None

        # Look for the <li> containing "Height" and extract the value
        for li in details_div.find_all("li"):
            if "Height" in li.get_text():
                span = li.find("span", itemprop="height")
                if span:
                    height_text = span.get_text(strip=True)
                    match = re.search(r"([\d,]+)\s*m", height_text)
                    if match:
                        height_m = match.group(1).replace(",", ".")
                        height_cm = int(float(height_m) * 100)
                        logging.info(f"{player_name}: Height: {height_cm} cm")
                        return height_cm

        logging.warning(f"{player_name}: Height not found")
        return None

    except Exception as e:
        logging.error(f"{player_name}: Error fetching height: {e}")
        return None
