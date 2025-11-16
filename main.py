from config import LEAGUE_CONFIG
from fbref_scraper import FBrefCrawler

if __name__ == "__main__":
    # Example 1: Scrape single league
    with FBrefCrawler(headless=True) as crawler:
        players = crawler.scrape_all_leagues(LEAGUE_CONFIG)
