from multiprocessing.pool import Pool
from config import LEAGUE_CONFIG, MAX_THREADS
from fbref_scraper import FBrefCrawler


def scrape_one_league(args):
    league_name, league_url = args
    crawler = FBrefCrawler(headless=True)

    csv_file = f"{league_name.replace(' ', '_')}.csv"
    json_file = f"{league_name.replace(' ', '_')}.json"

    crawler.scrape_league_streaming(
        league_name, league_url, csv_file=csv_file, json_file=json_file
    )

    crawler.close()
    return league_name


if __name__ == "__main__":
    leagues = list(LEAGUE_CONFIG.items())

    with Pool(MAX_THREADS) as p:
        result = p.map(scrape_one_league, leagues)

    print("DONE:", result)
