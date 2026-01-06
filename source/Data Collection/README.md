# Data Collection

**Purpose**

This directory contains scripts and resources used to collect raw football player data from various public sources (e.g. Transfermarkt). The outputs are raw CSV files stored in `data/` and may be used as input for subsequent preprocessing and analysis steps.

**Contents**

- `main.py` - orchestration script to run the collection pipeline (if provided).
- `DataCollection.ipynb` - exploratory notebook or pipeline examples for data collection.
- `scraper/` - contains scraping modules (e.g. `football_players_scraper.py`, `transfermarkt_scraper.py`).
- `data/` - CSV files collected from sources (raw data).

**Requirements & Installation**

Install the Python dependencies listed in `requirements.txt` (already present in this folder):

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

**How to run**

- Run individual scrapers: `python scraper/football_players_scraper.py` (check script arguments)
- Or run `main.py` if it orchestrates scraping pipelines.

**Notes**

- Use a separate virtual environment for scraping to avoid dependency conflicts.
- Some scrapers may require a Selenium WebDriver (e.g. `webdriver-manager` handles this automatically).
- Respect target sites' robots.txt and terms of service when scraping.
