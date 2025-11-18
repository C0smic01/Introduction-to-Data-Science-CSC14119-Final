HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ============================================================================
# DATA SCHEMA
# ============================================================================
BASE_SCHEMA = {
    "player_id": None,
    "player_name": "",
    "age": None,
    "nationality": "",
    "height": None,
    "foot": "",
    "position": "",
    "current_club": "",
    "league": "",
    "market_value": None,
    "appearances": 0,
    "minutes_played": 0,
    "minutes_per_game": 0.0,
    "goals": 0,
    "assists": 0,
    "goals_per_90": 0.0,
    "assists_per_90": 0.0,
    "shots": 0,
    "shots_on_target": 0,
    "xG": 0.0,
    "xAG": 0.0,
    "key_passes": 0,
    "tackles": 0,
    "interceptions": 0,
    "clearances": 0,
    "aerial_wins": 0,
    "aerial_win_rate": 0.0,
    "clean_sheets": 0,
    "saves": 0,
    "save_percentage": 0.0,
    "goals_conceded": 0,
    "goals_conceded_per_90": 0.0,
    "psxg_minus_ga": 0.0,
    "passes_completed": 0,
    "pass_accuracy": 0.0,
    "progressive_passes": 0,
}

# ============================================================================
# FBREF DEFAULT LEAGUES
# ============================================================================
SEASON = "2024-2025"

LEAGUE_CONFIG = {
    "La Liga": f"https://fbref.com/en/comps/12/{SEASON}/{SEASON}-La-Liga-Stats",
    "Premier League": f"https://fbref.com/en/comps/9/{SEASON}/{SEASON}-Premier-League-Stats",
    "Serie A": f"https://fbref.com/en/comps/11/{SEASON}/{SEASON}-Serie-A-Stats",
    "Bundesliga": f"https://fbref.com/en/comps/20/{SEASON}/{SEASON}-Bundesliga-Stats",
    "Ligue 1": f"https://fbref.com/en/comps/13/{SEASON}/{SEASON}-Ligue-1-Stats",
    "Eredivisie": f"https://fbref.com/en/comps/23/{SEASON}/{SEASON}-Eredivisie-Stats",
    "Primeira Liga": f"https://fbref.com/en/comps/32/{SEASON}/{SEASON}-Primeira-Liga-Stats",
}

# League levels
LEAGUE_LEVELS = {
    "GB1": 1,
    "ES1": 1,
    "IT1": 1,
    "L1": 1,
    "FR1": 1,  # Top 5
    "NL1": 2,
    "PO1": 2,
    "BE1": 2,  # Second tier
}


# Delays to avoid being blocked
DELAY_BETWEEN_REQUESTS = 2  # seconds
DELAY_BETWEEN_PLAYERS = 1  # seconds


# URL
TRANSFERMARKT_QUICKSEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"


# Thread
MAX_THREADS = 1

# Delay
MIN_DELAY = 1.5
MAX_DELAY = 3.0
