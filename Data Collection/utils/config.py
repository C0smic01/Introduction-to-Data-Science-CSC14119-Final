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
    # ========================
    # BASIC INFO
    # ========================
    "player_id": None,
    "player_name": None,
    "age": None,
    "nationality": None,
    "height": None,
    "foot": None,
    "position": None,
    "current_club": None,
    "league": None,
    # ========================
    # PLAYING TIME
    # ========================
    "appearances": 0,
    "minutes_played": 0,
    "minutes_per_game": None,
    # ========================
    # OUTFIELD - ATTACKING (13 fields)
    # ========================
    "goals": 0,
    "assists": 0,
    "goals_per_90": None,
    "assists_per_90": None,
    "npg_per90": None,
    "npxg_per90": None,
    "xag_per90": None,
    "npxg_xag_per90": None,
    "xg_per90": None,
    "shots_per90": None,
    "shots_on_target_per90": None,
    "shots_on_target_pct": None,
    "avg_shot_distance": None,
    "sca_per90": None,
    "gca_per90": None,
    # ========================
    # OUTFIELD - PLAYMAKING (8 fields)
    # ========================
    "key_passes_per90": None,
    "passes_completed_per90": None,
    "pass_completion_pct": None,
    "passes_into_final_third_per90": None,
    "passes_into_penalty_area_per90": None,
    "progressive_passes_per90": None,
    "progressive_passes_rec_per90": None,
    "progressive_carries_per90": None,
    # ========================
    # OUTFIELD - BALL PROGRESSION (8 fields)
    # ========================
    "take_ons_per90": None,
    "take_on_success_pct": None,
    "carries_per90": None,
    "carries_into_final_third_per90": None,
    "touches_per90": None,
    "touches_att_third_per90": None,
    "touches_att_pen_per90": None,
    "passes_received_per90": None,
    # ========================
    # OUTFIELD - DEFENSIVE (4 fields)
    # ========================
    "tackles_per90": None,
    "interceptions_per90": None,
    "blocks_per90": None,
    "ball_recoveries_per90": None,
    # ========================
    # OUTFIELD - PHYSICAL (2 fields)
    # ========================
    "aerials_won_per90": None,
    "aerial_win_pct": None,
    # ========================
    # OUTFIELD - DISCIPLINE (3 fields)
    # ========================
    "yellow_cards_per90": None,
    "red_cards_per90": None,
    "fouls_committed_per90": None,
    # ========================
    # GOALKEEPER - SHOT STOPPING (8 fields)
    # ========================
    "goals_against_per90": None,
    "shots_on_target_against_per90": None,
    "saves_per90": None,
    "save_percentage": None,
    "clean_sheet_pct": None,
    "psxg_per_shot": None,
    "psxg_ga_per90": None,
    "penalty_save_pct": None,
    # ========================
    # GOALKEEPER - DISTRIBUTION (3 fields)
    # ========================
    "passes_attempted_per90": None,
    "launch_pct": None,
    "avg_pass_length": None,
    # ========================
    # GOALKEEPER - SWEEPING (2 fields)
    # ========================
    "def_actions_outside_pen_per90": None,
    "avg_distance_def_actions": None,
    # ========================
    # GOALKEEPER - CROSS HANDLING (1 field)
    # ========================
    "crosses_stopped_pct": None,
    # ========================
    # GOALKEEPER - MATCH CONTEXT (3 fields)
    # ========================
    "wins_per90": None,
    "draws_per90": None,
    "losses_per90": None,
    # ========================
    # TARGET
    # ========================
    "market_value": None,
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

# URL
TRANSFERMARKT_PLAYER_SEACH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"

# Thread
MAX_THREADS = 1

# Delay
MIN_DELAY = 1.5
MAX_DELAY = 3.0

DATA_FOLDER = "data"
CSV_FOLDER = f"{DATA_FOLDER}/csv"
JSON_FOLDER = f"{DATA_FOLDER}/json"
HEADLESS = True
