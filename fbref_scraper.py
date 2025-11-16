"""
FBref Player Statistics Crawler
Modular class for scraping player data from FBref.com
Can be used standalone or combined with other crawlers
"""

import time
import random
import logging
import csv
import json
import hashlib
import unicodedata
import re
from datetime import datetime
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from config import BASE_SCHEMA, LEAGUE_CONFIG
from transfermarkt_scraper import get_market_value

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def generate_player_id(
    player_name: str, dob: Optional[str] = None, nationality: Optional[str] = None
) -> str:
    """
    Generate unique player ID from name, DOB, and nationality
    Format: firstname-lastname-hash6
    Example: lionel-messi-a3f8e2
    """
    name = unicodedata.normalize("NFKD", player_name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[-\s]+", "-", name).strip("-")

    hash_input = f"{player_name}:{dob or ''}:{nationality or ''}"
    hash_obj = hashlib.md5(hash_input.encode("utf-8"))
    hash_suffix = hash_obj.hexdigest()[:6]

    return f"{name}-{hash_suffix}"


def random_delay(a: float = 1.5, b: float = 3.0) -> None:
    """Add random delay to avoid rate limiting"""
    time.sleep(random.uniform(a, b))


def clean_number(val, allow_float: bool = True):
    """Convert string value to number, handling various formats"""
    if val is None:
        return 0
    try:
        s = str(val).strip().replace(",", "").replace("%", "")
        if s == "":
            return 0
        if allow_float and (
            "." in s or s.replace(".", "", 1).replace("-", "", 1).isdigit()
        ):
            return float(s)
        return int(float(s))
    except Exception:
        return 0


def find_table_in_comments(
    soup: BeautifulSoup, needle: Optional[str] = None, id_contains: Optional[str] = None
):
    """Find HTML table hidden in comments (FBref's anti-scraping technique)"""
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if needle and needle not in c:
            continue
        try:
            s2 = BeautifulSoup(c, "html.parser")
            if id_contains:
                t = s2.find("table", id=lambda x: x and id_contains in x)
            else:
                t = s2.find("table")
            if t:
                return t
        except Exception:
            continue
    return None


# ============================================================================
# MAIN CRAWLER CLASS
# ============================================================================


class FBrefCrawler:
    """
    Scraper for FBref.com player statistics

    Usage:
        crawler = FBrefCrawler(headless=True)
        players = crawler.scrape_league("La Liga", "https://fbref.com/...")
        crawler.close()
    """

    def __init__(self, headless: bool = True, user_agent: Optional[str] = None):
        """
        Initialize crawler with Chrome WebDriver

        Args:
            headless: Run browser in headless mode
            user_agent: Custom user agent string
        """
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        ua = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        chrome_options.add_argument(f"user-agent={ua}")

        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )
        self.players: List[Dict] = []
        self.seen_ids: Set[str] = set()

    def close(self) -> None:
        """Close the browser driver"""
        try:
            self.driver.quit()
        except:
            pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()

    # ------------------------------------------------------------------------
    # CORE SCRAPING METHODS
    # ------------------------------------------------------------------------

    def get_page_soup(self, url: str, wait: float = 1.5) -> Optional[BeautifulSoup]:
        """Fetch page and return BeautifulSoup object"""
        logging.info(f"GET {url}")
        try:
            self.driver.get(url)
        except Exception as e:
            logging.warning(f"Failed to load {url}: {e}")
            return None
        random_delay(wait, wait + 1.0)
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def get_league_clubs(self, league_url: str) -> List[Dict[str, str]]:
        """
        Extract all clubs from a league page

        Returns:
            List of dicts with 'club_name' and 'club_url'
        """
        soup = self.get_page_soup(league_url)
        if not soup:
            return []

        table = soup.find("table", class_="stats_table")
        if not table:
            table = find_table_in_comments(soup, needle="standings")

        clubs = []
        if table:
            tbody = table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    team_cell = row.find("td", {"data-stat": "team"})
                    if team_cell:
                        a = team_cell.find("a")
                        if a and a.get("href"):
                            clubs.append(
                                {
                                    "club_name": a.get_text(strip=True),
                                    "club_url": "https://fbref.com" + a["href"],
                                }
                            )

        logging.info(f"Found {len(clubs)} clubs")
        return clubs

    def get_club_players(self, club_url: str) -> List[Dict]:
        """
        Extract basic player info from club page

        Returns:
            List of dicts with player_name, player_url, and basic stats
        """
        soup = self.get_page_soup(club_url)
        if not soup:
            return []

        table = soup.find("table", id=lambda v: v and v.startswith("stats_standard"))
        if not table:
            table = find_table_in_comments(soup, needle="stats_standard")

        players = []
        if table:
            tbody = table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class"):
                        continue

                    th = row.find("th", {"data-stat": "player"})
                    if not th:
                        continue

                    a = th.find("a")
                    if not a or not a.get("href"):
                        continue

                    def get_stat(col, allow_float=True):
                        td = row.find("td", {"data-stat": col})
                        return clean_number(
                            td.text if td else None, allow_float=allow_float
                        )

                    players.append(
                        {
                            "player_name": a.get_text(strip=True),
                            "player_url": "https://fbref.com" + a["href"],
                            "appearances": get_stat("games", allow_float=False),
                            "minutes_played": get_stat("minutes", allow_float=False),
                            "goals": get_stat("goals", allow_float=False),
                            "assists": get_stat("assists", allow_float=False),
                            "xG": get_stat("xg"),
                            "xAG": get_stat("xg_assist"),
                        }
                    )

        logging.info(f"Found {len(players)} players")
        return players

    def scrape_player_full(
        self,
        player_url: str,
        league_name: Optional[str] = None,
        club_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Scrape complete player profile with all statistics
        """
        soup = self.get_page_soup(player_url)
        if not soup:
            return None

        stats = dict(BASE_SCHEMA)
        stats["league"] = league_name or ""
        stats["current_club"] = club_name or ""

        # Lấy container chính
        info_div = soup.find("div", id="info")
        if not info_div:
            return stats

        # --- Player name ---
        h1 = info_div.find("h1")
        if h1:
            stats["player_name"] = h1.get_text(strip=True)

        # --- Nationality ---
        nat_link = info_div.find("a", href=lambda x: x and "/country/" in x)
        if nat_link:
            stats["nationality"] = nat_link.get_text(strip=True)

        # --- Date of birth & age ---
        dob = None
        birth_span = info_div.find("span", id="necro-birth")
        if birth_span and birth_span.get("data-birth"):
            dob = birth_span["data-birth"]
            try:
                dt = datetime.strptime(dob, "%Y-%m-%d")
                today = datetime.today()
                stats["age"] = (
                    today.year
                    - dt.year
                    - ((today.month, today.day) < (dt.month, dt.day))
                )
            except:
                pass

        # --- Generate unique player ID ---
        player_id = generate_player_id(
            stats.get("player_name"), dob=dob, nationality=stats.get("nationality")
        )
        counter = 1
        base_id = player_id
        while player_id in self.seen_ids:
            player_id = f"{base_id}-{counter}"
            counter += 1
        self.seen_ids.add(player_id)
        stats["player_id"] = player_id

        # --- Height ---
        height_span = info_div.find("span", string=lambda s: s and s.endswith("cm"))
        if height_span:
            try:
                stats["height"] = float(
                    height_span.get_text(strip=True).replace("cm", "").strip()
                )
            except:
                pass

        # --- Position & Footed ---
        info_text = info_div.get_text(" ", strip=True).replace("\xa0", " ")
        # Dùng regex để tìm Position và Footed
        m_pos = re.search(r"Position:\s*([A-Za-z0-9\-]+)", info_text)
        m_foot = re.search(r"Footed:\s*([A-Za-z]+)", info_text)

        if m_pos:
            stats["position"] = m_pos.group(1).strip()
        if m_foot:
            stats["foot"] = m_foot.group(1).strip()

        # --- Parse stats tables ---
        self._parse_standard_stats(soup, stats)
        self._parse_defensive_stats(soup, stats)
        self._parse_passing_stats(soup, stats)
        self._parse_goalkeeper_stats(soup, stats)
        self._calculate_derived_fields(stats)

        # --- Market value ---
        try:
            stats["market_value"] = get_market_value(player_name=stats["player_name"])
        except Exception as e:
            logging.warning(
                f"Can not get market value for {stats.get('player_name')}: {e}"
            )
            stats["market_value"] = None

        return stats

    # ------------------------------------------------------------------------
    # TABLE PARSING HELPERS
    # ------------------------------------------------------------------------

    def _parse_standard_stats(self, soup: BeautifulSoup, stats: Dict) -> None:
        """Parse standard statistics table"""
        std_table = soup.find(
            "table", id=lambda x: x and x.startswith("stats_standard")
        )
        if not std_table:
            std_table = find_table_in_comments(soup, needle="stats_standard")

        if std_table:
            tbody = std_table.find("tbody")
            if tbody:
                agg = {}
                mapping = {
                    "games": "appearances",
                    "minutes": "minutes_played",
                    "goals": "goals",
                    "assists": "assists",
                    "shots": "shots",
                    "shots_on_target": "shots_on_target",
                    "xg": "xG",
                    "xg_assist": "xAG",
                    "passes_completed": "passes_completed",
                }

                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class"):
                        continue

                    for td in row.find_all("td"):
                        dstat = td.get("data-stat")
                        if dstat in mapping:
                            key = mapping[dstat]
                            is_int = key in [
                                "appearances",
                                "minutes_played",
                                "goals",
                                "assists",
                                "shots",
                                "shots_on_target",
                                "passes_completed",
                            ]
                            val = clean_number(
                                td.get_text(strip=True), allow_float=not is_int
                            )
                            agg[key] = agg.get(key, 0) + val

                for k, v in agg.items():
                    stats[k] = v

    def _parse_defensive_stats(self, soup: BeautifulSoup, stats: Dict) -> None:
        """Parse defensive statistics table"""
        def_table = soup.find("table", id=lambda x: x and "defense" in x)
        if not def_table:
            def_table = find_table_in_comments(soup, needle="Defense")

        if def_table:
            tbody = def_table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class"):
                        continue

                    def get_stat(dstat, allow_float=False):
                        td = row.find("td", {"data-stat": dstat})
                        return clean_number(
                            td.get_text(strip=True) if td else None,
                            allow_float=allow_float,
                        )

                    if val := get_stat("tackles"):
                        stats["tackles"] = val
                    if val := get_stat("interceptions"):
                        stats["interceptions"] = val
                    if val := get_stat("clearances"):
                        stats["clearances"] = val
                    if val := get_stat("aerials_won"):
                        stats["aerial_wins"] = val
                    if val := get_stat("aerials_won_pct", allow_float=True):
                        stats["aerial_win_rate"] = val

    def _parse_passing_stats(self, soup: BeautifulSoup, stats: Dict) -> None:
        """Parse passing statistics table"""
        pass_table = soup.find("table", id=lambda x: x and "passing" in x)
        if not pass_table:
            pass_table = find_table_in_comments(soup, needle="Passes")

        if pass_table:
            tbody = pass_table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class"):
                        continue

                    def get_stat(dstat, allow_float=False):
                        td = row.find("td", {"data-stat": dstat})
                        return clean_number(
                            td.get_text(strip=True) if td else None,
                            allow_float=allow_float,
                        )

                    if val := get_stat("passes_completed"):
                        stats["passes_completed"] = val
                    if val := get_stat("passes_pct", allow_float=True):
                        stats["pass_accuracy"] = val
                    if val := get_stat("progressive_passes"):
                        stats["progressive_passes"] = val
                    if val := get_stat("passes_into_final_third"):
                        stats["key_passes"] = val

    def _parse_goalkeeper_stats(self, soup: BeautifulSoup, stats: Dict) -> None:
        """Parse goalkeeper statistics table"""
        gk_table = soup.find("table", id=lambda x: x and "keeper" in x)
        if not gk_table:
            gk_table = find_table_in_comments(soup, needle="Goalkeeping")

        if gk_table:
            tbody = gk_table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class"):
                        continue

                    def get_stat(dstat, allow_float=False):
                        td = row.find("td", {"data-stat": dstat})
                        return clean_number(
                            td.get_text(strip=True) if td else None,
                            allow_float=allow_float,
                        )

                    if val := get_stat("gk_saves"):
                        stats["saves"] = val
                    if val := get_stat("gk_save_pct", allow_float=True):
                        stats["save_percentage"] = val
                    if val := get_stat("gk_goals_against"):
                        stats["goals_conceded"] = val
                    if val := get_stat("gk_clean_sheets"):
                        stats["clean_sheets"] = val
                    if val := get_stat("gk_psxg_gk", allow_float=True):
                        stats["psxg_minus_ga"] = val

    def _calculate_derived_fields(self, stats: Dict) -> None:
        """Calculate per-90 and other derived statistics"""
        if stats["minutes_played"] and stats["appearances"]:
            stats["minutes_per_game"] = round(
                stats["minutes_played"] / max(1, stats["appearances"]), 1
            )

        if stats["minutes_played"] > 0:
            mins_90 = stats["minutes_played"] / 90
            stats["goals_per_90"] = round(stats["goals"] / mins_90, 2)
            stats["assists_per_90"] = round(stats["assists"] / mins_90, 2)

            if stats["goals_conceded"]:
                stats["goals_conceded_per_90"] = round(
                    stats["goals_conceded"] / mins_90, 2
                )

    # ------------------------------------------------------------------------
    # HIGH-LEVEL SCRAPING METHODS
    # ------------------------------------------------------------------------

    def scrape_league(self, league_name: str, league_url: str) -> List[Dict]:
        """
        Scrape all players from a league

        Args:
            league_name: Name of the league
            league_url: FBref URL for the league

        Returns:
            List of player dicts
        """
        logging.info(f"Starting league: {league_name}")
        clubs = self.get_league_clubs(league_url)
        league_players = []

        for club in clubs:
            logging.info(f"Scraping club: {club['club_name']}")
            club_players = self.get_club_players(club["club_url"])

            for p in club_players:
                try:
                    full = self.scrape_player_full(
                        p["player_url"], league_name, club["club_name"]
                    )

                    if not full:
                        continue

                    # Merge basic club stats
                    full["appearances"] = p.get("appearances") or full.get(
                        "appearances", 0
                    )
                    full["minutes_played"] = p.get("minutes_played") or full.get(
                        "minutes_played", 0
                    )
                    full["goals"] = p.get("goals") or full.get("goals", 0)
                    full["assists"] = p.get("assists") or full.get("assists", 0)
                    full["xG"] = p.get("xG") or full.get("xG", 0)
                    full["xAG"] = p.get("xAG") or full.get("xAG", 0)

                    # Recalculate derived fields
                    self._calculate_derived_fields(full)

                    league_players.append(full)
                    self.players.append(full)

                    logging.info(f"✓ {full['player_name']} (ID: {full['player_id']})")

                except Exception as e:
                    logging.exception(f"✗ Error scraping {p.get('player_name')}: {e}")

        logging.info(
            f"League {league_name} complete. Total: {len(league_players)} players"
        )
        return league_players

    def scrape_all_leagues(
        self, leagues: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Scrape all configured leagues

        Args:
            leagues: Dict of league_name: league_url. Uses DEFAULT_LEAGUES if None

        Returns:
            List of all player dicts
        """
        leagues = leagues or self.DEFAULT_LEAGUES

        for name, url in leagues.items():
            self.scrape_league_streaming(name, url)

        logging.info(f"✓ All leagues complete. Total: {len(self.players)} players")
        return self.players

    # ------------------------------------------------------------------------
    # DATA EXPORT METHODS
    # ------------------------------------------------------------------------

    def save_to_csv(self, filename: str = "fbref_players.csv") -> None:
        """Save all scraped players to CSV"""
        if not self.players:
            logging.warning("No players to save")
            return

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(BASE_SCHEMA.keys()))
            writer.writeheader()

            for player in self.players:
                row = {k: player.get(k, BASE_SCHEMA[k]) for k in BASE_SCHEMA.keys()}
                writer.writerow(row)

        logging.info(f"✓ Saved {len(self.players)} players to {filename}")

    def save_to_json(self, filename: str = "fbref_players.json") -> None:
        """Save all scraped players to JSON"""
        if not self.players:
            logging.warning("No players to save")
            return

        clean_players = [
            {k: p.get(k, BASE_SCHEMA[k]) for k in BASE_SCHEMA.keys()}
            for p in self.players
        ]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(clean_players, f, ensure_ascii=False, indent=2)

        logging.info(f"✓ Saved {len(self.players)} players to {filename}")

    def get_player_by_id(self, player_id: str) -> Optional[Dict]:
        """Get player data by ID"""
        for player in self.players:
            if player.get("player_id") == player_id:
                return player
        return None

    def get_players_by_club(self, club_name: str) -> List[Dict]:
        """Get all players from a specific club"""
        return [p for p in self.players if p.get("current_club") == club_name]

    def get_players_by_league(self, league_name: str) -> List[Dict]:
        """Get all players from a specific league"""
        return [p for p in self.players if p.get("league") == league_name]

    def scrape_league_streaming(
        self,
        league_name: str,
        league_url: str,
        csv_file="players.csv",
        json_file="players.json",
    ) -> None:
        """
        Scrape all players in a league and save immediately to CSV/JSON
        """
        logging.info(f"Starting league: {league_name}")
        clubs = self.get_league_clubs(league_url)

        # Open CSV and JSON files once
        csv_f = open(csv_file, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_f, fieldnames=list(BASE_SCHEMA.keys()))
        csv_writer.writeheader()

        json_f = open(json_file, "w", encoding="utf-8")
        json_f.write("[\n")  # start JSON array
        first = True  # flag for JSON commas

        try:
            for club in clubs:
                logging.info(f"Scraping club: {club['club_name']}")
                club_players = self.get_club_players(club["club_url"])

                for p in club_players:
                    try:
                        full = self.scrape_player_full(
                            p["player_url"], league_name, club["club_name"]
                        )
                        if not full:
                            continue

                        # Merge basic stats
                        for stat in [
                            "appearances",
                            "minutes_played",
                            "goals",
                            "assists",
                            "xG",
                            "xAG",
                        ]:
                            full[stat] = p.get(stat) or full.get(stat, 0)

                        self._calculate_derived_fields(full)

                        # --- Write CSV ---
                        csv_writer.writerow(
                            {k: full.get(k, BASE_SCHEMA[k]) for k in BASE_SCHEMA.keys()}
                        )
                        csv_f.flush()

                        # --- Write JSON ---
                        if not first:
                            json_f.write(",\n")
                        else:
                            first = False
                        json.dump(
                            {
                                k: full.get(k, BASE_SCHEMA[k])
                                for k in BASE_SCHEMA.keys()
                            },
                            json_f,
                            ensure_ascii=False,
                            indent=2,
                        )
                        json_f.flush()

                        logging.info(
                            f"✓ {full['player_name']} (ID: {full['player_id']})"
                        )

                    except Exception as e:
                        logging.exception(
                            f"✗ Error scraping {p.get('player_name')}: {e}"
                        )

        finally:
            csv_f.close()
            json_f.write("\n]")  # close JSON array
            json_f.close()

        logging.info(f"League {league_name} complete.")
