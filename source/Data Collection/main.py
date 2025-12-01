from utils.imports import *
from utils.config import CSV_FOLDER, HEADLESS, JSON_FOLDER, LEAGUE_CONFIG, MAX_THREADS
from scraper.football_players_scraper import FootbalPlayerCrawler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def scrape_one_league(args):
    league_name, league_url = args
    crawler = FootbalPlayerCrawler(headless=HEADLESS)

    os.makedirs(CSV_FOLDER, exist_ok=True)
    os.makedirs(JSON_FOLDER, exist_ok=True)

    csv_file = os.path.join(CSV_FOLDER, f"{league_name.replace(' ', '_')}.csv")
    json_file = os.path.join(JSON_FOLDER, f"{league_name.replace(' ', '_')}.json")

    crawler.scrape_league(
        league_name, league_url, csv_file=csv_file, json_file=json_file
    )

    crawler.close()
    return league_name


if __name__ == "__main__":
    leagues = list(LEAGUE_CONFIG.items())

    with Pool(MAX_THREADS) as p:
        result = p.map(scrape_one_league, leagues)

    print("DONE:", result)
