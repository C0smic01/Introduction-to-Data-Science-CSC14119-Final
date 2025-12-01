"""
FBref Player Statistics Crawler
Modular class for scraping player data from FBref.com
Can be used standalone or combined with other crawlers
"""

from utils.imports import *
from utils.config import (
    BASE_SCHEMA,
    OUTFIELD_SCOUTING_MAPPING,
    OUTFIELD_DETAILED_MAPPING,
    GOALKEEPER_SCOUTING_MAPPING,
    GOALKEEPER_DETAILED_MAPPING,
)
from utils.utils import generate_player_id, clean_number, find_table_in_comments
from scraper.transfermarkt_scraper import (
    get_height,
    get_market_value,
    get_profile_url,
    random_delay,
)


class FootbalPlayerCrawler:
    """
    FBref Crawler class for scraping player statistics.

    Usage:
        crawler = FootbalPlayerCrawler(headless=True)
        crawler.scrape_league_streaming("La Liga", "https://fbref.com/...")
        crawler.close()
    """

    def __init__(self, headless: bool = True, user_agent: Optional[str] = None):
        """Initialize crawler with Chrome WebDriver."""
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
        self.seen_ids: Set[str] = set()

    def close(self) -> None:
        """Close the Selenium WebDriver."""
        try:
            self.driver.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------------
    # CORE SCRAPING METHODS
    # ------------------------------------------------------------------------

    def get_page_soup(self, url: str, wait: float = 1.5) -> Optional[BeautifulSoup]:
        """Fetch a page and return its BeautifulSoup object."""
        logging.info(f"GET {url}")
        try:
            self.driver.get(url)
        except Exception as e:
            logging.warning(f"Failed to load {url}: {e}")
            return None
        random_delay(wait, wait + 1.0)
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def get_league_clubs(self, league_url: str) -> List[Dict[str, str]]:
        """Extract all clubs from a league page."""
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
        """Extract basic player info from club page."""
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

    def get_scouting_report_url(self, profile_soup: BeautifulSoup) -> Optional[str]:
        """
        Extract the scouting report URL from player profile page.
        Looks for link like: /en/players/xxx/scout/365_m1/Player-Name-Scouting-Report
        """
        for link in profile_soup.find_all("a", href=True):
            href = link.get("href", "")
            if "/scout/" in href and "Scouting-Report" in href:
                full_url = "https://fbref.com" + href
                logging.info(f"Found scouting report URL: {full_url}")
                return full_url

        logging.warning("No scouting report link found in player profile")
        return None

    def scrape_full_players_data(
        self,
        player_url: str,
        league_name: Optional[str] = None,
        club_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """Scrape complete player profile with all statistics including scouting report."""
        soup = self.get_page_soup(player_url)
        if not soup:
            logging.warning(f"Cannot load page: {player_url}")
            return None

        stats = dict(BASE_SCHEMA)
        stats["league"] = league_name or None
        stats["current_club"] = club_name or None

        info_div = soup.find("div", id="info")
        if not info_div:
            logging.warning(f"No info div found for {player_url}")
            return None

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
                stats["height"] = None
        else:
            if stats.get("player_name") and stats.get("current_club"):
                try:
                    profile_url = get_profile_url(self.driver, stats["player_name"])
                    if profile_url:
                        height_cm = get_height(profile_url, stats["player_name"])
                        stats["height"] = height_cm
                except Exception as e:
                    logging.warning(
                        f"{stats['player_name']}: Cannot get height from fallback: {e}"
                    )

        # --- Position & Footed ---
        info_text = info_div.get_text(" ", strip=True).replace("\xa0", " ")
        m_pos = re.search(r"Position:\s*([A-Za-z0-9\-]+)", info_text)
        m_foot = re.search(r"Footed:\s*([A-Za-z]+)", info_text)
        if m_pos:
            stats["position"] = m_pos.group(1).strip()
        if m_foot:
            stats["foot"] = m_foot.group(1).strip()

        # --- Get scouting report URL and scrape it ---
        scouting_url = self.get_scouting_report_url(soup)
        if scouting_url:
            logging.info(f"Parsing scouting report for {stats.get('player_name')}")
            has_data = self._parse_full_scouting_report(scouting_url, stats)

            # Bỏ qua cầu thủ nếu không tìm thấy table chứa data
            if not has_data:
                logging.warning(
                    f"No scouting data tables found for {stats.get('player_name')}, skipping player"
                )
                return None
        else:
            logging.warning(
                f"No scouting report for {stats.get('player_name')}, skipping player"
            )
            return None

        # --- Calculate derived fields ---
        self._calculate_derived_fields(stats)
        self._calculate_totals_from_per90(stats)

        # --- Market value ---
        try:
            stats["market_value"] = get_market_value(player_name=stats["player_name"])
        except Exception as e:
            logging.warning(
                f"Cannot get market value for {stats.get('player_name')}: {e}"
            )

        return stats

    def _parse_full_scouting_report(self, scouting_url: str, stats: Dict) -> bool:
        """
        Parse the complete scouting report page.
        Returns True if data tables were found, False otherwise.
        """
        soup = self.get_page_soup(scouting_url)
        if not soup:
            logging.warning("Failed to load scouting report page")
            return False

        # Check if player is a goalkeeper
        is_goalkeeper = stats.get("position", "").upper() == "GK"

        if is_goalkeeper:
            has_data = self._parse_gk_full_scouting(soup, stats)
        else:
            has_data = self._parse_outfield_full_scouting(soup, stats)

        return has_data

    def _parse_outfield_full_scouting(self, soup: BeautifulSoup, stats: Dict) -> bool:
        """
        Parse full scouting report for outfield players from scout_full tables.
        Returns True if data was found, False otherwise.
        """
        tables = soup.find_all("table", id=lambda x: x and "scout_full" in x)

        if not tables:
            logging.debug("No scout_full tables found, trying comments...")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                try:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    found_tables = comment_soup.find_all(
                        "table", id=lambda x: x and "scout_full" in x
                    )
                    tables.extend(found_tables)
                except:
                    continue

        if not tables:
            logging.warning("No scout_full tables found for outfield player")
            return False

        combined_mapping = {**OUTFIELD_SCOUTING_MAPPING, **OUTFIELD_DETAILED_MAPPING}
        data_found = False

        for table in tables:
            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                stat_th = row.find("th", {"data-stat": "statistic"})
                if not stat_th:
                    continue

                stat_name = stat_th.get_text(strip=True)
                per90_td = row.find("td", {"data-stat": "per90"})
                if not per90_td:
                    continue

                per90_value = clean_number(
                    per90_td.get_text(strip=True), allow_float=True
                )

                if stat_name in combined_mapping:
                    field_name = combined_mapping[stat_name]
                    stats[field_name] = per90_value
                    logging.debug(f"  {stat_name}: {per90_value} -> {field_name}")
                    data_found = True

        return data_found

    def _parse_gk_full_scouting(self, soup: BeautifulSoup, stats: Dict) -> bool:
        """
        Parse full scouting report for goalkeepers from scout_full tables.
        Returns True if data was found, False otherwise.
        """
        tables = soup.find_all("table", id=lambda x: x and "scout_full" in x)

        if not tables:
            logging.debug("No scout_full tables found, trying comments...")
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                try:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    found_tables = comment_soup.find_all(
                        "table", id=lambda x: x and "scout_full" in x
                    )
                    tables.extend(found_tables)
                except:
                    continue

        if not tables:
            logging.warning("No scout_full tables found for goalkeeper")
            return False

        combined_mapping = {
            **GOALKEEPER_SCOUTING_MAPPING,
            **GOALKEEPER_DETAILED_MAPPING,
        }
        data_found = False

        for table in tables:
            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                stat_th = row.find("th", {"data-stat": "statistic"})
                if not stat_th:
                    continue

                stat_name = stat_th.get_text(strip=True)
                per90_td = row.find("td", {"data-stat": "per90"})
                if not per90_td:
                    continue

                per90_text = per90_td.get_text(strip=True)

                # Handle +/- values
                if per90_text.startswith("+"):
                    per90_value = clean_number(per90_text[1:], allow_float=True)
                elif per90_text.startswith("-"):
                    per90_value = (
                        -clean_number(per90_text[1:], allow_float=True)
                        if clean_number(per90_text[1:], allow_float=True)
                        else None
                    )
                else:
                    per90_value = clean_number(per90_text, allow_float=True)

                if stat_name in combined_mapping:
                    field_name = combined_mapping[stat_name]
                    stats[field_name] = per90_value
                    logging.debug(f"  {stat_name}: {per90_value} -> {field_name}")
                    data_found = True

        return data_found

    def _calculate_derived_fields(self, stats: Dict) -> None:
        """Calculate per-90 and other derived statistics."""
        if stats.get("minutes_played") and stats.get("appearances"):
            stats["minutes_per_game"] = round(
                stats["minutes_played"] / max(1, stats["appearances"]), 1
            )

        if stats.get("minutes_played") and stats["minutes_played"] > 0:
            mins_90 = stats["minutes_played"] / 90

            if stats.get("goals") and stats["goals"] > 0:
                stats["goals_per_90"] = round(stats["goals"] / mins_90, 2)

            if stats.get("assists") and stats["assists"] > 0:
                stats["assists_per_90"] = round(stats["assists"] / mins_90, 2)

            if stats.get("goals_conceded") and stats["goals_conceded"] > 0:
                stats["goals_conceded_per_90"] = round(
                    stats["goals_conceded"] / mins_90, 2
                )

    def _calculate_totals_from_per90(self, stats: Dict) -> None:
        """Calculate total values from per_90 stats when totals are missing."""
        if not stats.get("minutes_played") or stats["minutes_played"] <= 0:
            return

        mins_90 = stats["minutes_played"] / 90

        # Only for outfield players
        if stats.get("position", "").upper() != "GK":
            mappings = [
                ("shots_per90", "shots", True),
                ("tackles_per90", "tackles", True),
                ("interceptions_per90", "interceptions", True),
                ("blocks_per90", "blocks", True),
                ("clearances_per90", "clearances", True),
                ("aerials_won_per90", "aerial_wins", True),
                ("progressive_passes_per90", "progressive_passes", True),
            ]

            for per90_field, total_field, is_integer in mappings:
                per90_val = stats.get(per90_field)
                if (
                    not stats.get(total_field) or stats.get(total_field) == 0
                ) and per90_val is not None:
                    calculated = per90_val * mins_90
                    stats[total_field] = (
                        int(round(calculated)) if is_integer else round(calculated, 2)
                    )

        # Goalkeeper specific
        else:
            if stats.get("saves_per90") is not None and not stats.get("saves"):
                stats["saves"] = int(round(stats["saves_per90"] * mins_90))

            if (
                stats.get("clean_sheet_pct") is not None
                and not stats.get("clean_sheets")
                and stats.get("appearances")
            ):
                stats["clean_sheets"] = int(
                    round(stats["appearances"] * stats["clean_sheet_pct"] / 100)
                )

    # ------------------------------------------------------------------------
    # DATA EXPORT
    # ------------------------------------------------------------------------

    def scrape_league(
        self,
        league_name: str,
        league_url: str,
        csv_file=None,
        json_file=None,
    ) -> None:
        """Scrape all players in a league and save immediately to CSV/JSON."""
        if csv_file is None:
            csv_file = f"{league_name.replace(' ', '_').lower()}_players.csv"
        if json_file is None:
            json_file = f"{league_name.replace(' ', '_').lower()}_players.json"

        logging.info(f"Starting league: {league_name}")
        logging.info(f"CSV file:  {csv_file}")
        logging.info(f"JSON file: {json_file}")

        clubs = self.get_league_clubs(league_url)

        csv_f = open(csv_file, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_f, fieldnames=list(BASE_SCHEMA.keys()))
        csv_writer.writeheader()

        json_f = open(json_file, "w", encoding="utf-8")
        json_f.write("[\n")
        first = True

        try:
            for club in clubs:
                logging.info(f"Scraping club: {club['club_name']}")
                club_players = self.get_club_players(club["club_url"])

                for p in club_players:
                    try:
                        full = self.scrape_full_players_data(
                            p["player_url"], league_name, club["club_name"]
                        )
                        if not full:
                            continue

                        # Merge basic stats from club page
                        for stat in [
                            "appearances",
                            "minutes_played",
                            "goals",
                            "assists",
                            "xG",
                            "xAG",
                        ]:
                            if p.get(stat) is not None:
                                full[stat] = p[stat]

                        self._calculate_derived_fields(full)
                        self._calculate_totals_from_per90(full)

                        # Write CSV
                        csv_writer.writerow(
                            {k: full.get(k, BASE_SCHEMA[k]) for k in BASE_SCHEMA.keys()}
                        )
                        csv_f.flush()

                        # Write JSON
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
            json_f.write("\n]")
            json_f.close()

        logging.info(f"League {league_name} complete.")
