import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import logging
import time
import random
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Thread-%(thread)d] - %(levelname)s - %(message)s'
)

# Constants
TRANSFERMARKT_PLAYER_SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name}"
CHECKPOINT_FILE = "scraping_checkpoint.json"

# Thread-safe locks
file_lock = Lock()
checkpoint_lock = Lock()

def random_delay(min_delay=2, max_delay=4):
    """Random delay to avoid overwhelming the server"""
    time.sleep(random.uniform(min_delay, max_delay))

def parse_market_value(market_value_str):
    """
    Parse market value string like '€3.50m' or '€600k' to float in millions
    """
    try:
        market_value_str = market_value_str.strip().replace('€', '').replace(',', '.')
        
        if 'm' in market_value_str.lower():
            return float(market_value_str.lower().replace('m', ''))
        elif 'k' in market_value_str.lower():
            return float(market_value_str.lower().replace('k', '')) / 1000
        else:
            return None
    except (ValueError, AttributeError):
        return None

def load_checkpoint():
    """Load checkpoint from file (thread-safe)"""
    with checkpoint_lock:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint = json.load(f)
                    logging.info(f"Checkpoint loaded: {len(checkpoint.get('processed_indices', []))} players already processed")
                    return checkpoint
            except Exception as e:
                logging.warning(f"Could not load checkpoint: {e}")
        return {'processed_indices': set(), 'updated_count': 0, 'failed_count': 0}

def save_checkpoint(checkpoint):
    """Save checkpoint to file (thread-safe)"""
    with checkpoint_lock:
        try:
            # Convert set to list for JSON serialization
            checkpoint_copy = checkpoint.copy()
            checkpoint_copy['processed_indices'] = list(checkpoint_copy['processed_indices'])
            checkpoint_copy['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_copy, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save checkpoint: {e}")

def delete_checkpoint():
    """Delete checkpoint file after completion"""
    with checkpoint_lock:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                os.remove(CHECKPOINT_FILE)
                logging.info("Checkpoint file deleted")
            except Exception as e:
                logging.warning(f"Could not delete checkpoint: {e}")

def setup_driver():
    """Setup Chrome driver with options"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    return driver

def get_market_value(driver, player_name: str, club_name: str = None, age: int = None) -> float | None:
    """
    Fetch the market value directly from search results table
    """
    url = TRANSFERMARKT_PLAYER_SEARCH_URL.format(player_name=player_name)
    
    try:
        driver.get(url)
        random_delay(1, 2)  # Shorter delay for multi-threading
        
        # Wait for search results table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "items"))
        )
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table", class_="items")
        
        if not table or not table.tbody:
            return None

        # Loop through all rows to find the matching player
        for row in table.tbody.find_all("tr"):
            # Extract player name
            name_link = row.find("a", title=True, href=lambda x: x and "/profil/spieler/" in x)
            if not name_link:
                continue
                
            row_player_name = name_link.get("title", "").strip()
            
            # Check if name matches (case-insensitive)
            if row_player_name.lower() != player_name.lower():
                continue
            
            # Check club if provided
            if club_name:
                club_td = row.find("td", class_="inline-table")
                if club_td:
                    club_links = club_td.find_all("a")
                    if club_links:
                        row_club_name = club_links[-1].get("title", "").strip()
                        if club_name.lower() not in row_club_name.lower():
                            continue
            
            # Check age if provided
            if age is not None:
                age_cells = row.find_all("td", class_="zentriert")
                if len(age_cells) >= 3:
                    try:
                        row_age = int(age_cells[2].get_text(strip=True))
                        if row_age != age:
                            continue
                    except (ValueError, AttributeError):
                        continue
            
            # Found matching player - get market value
            mv_cell = row.find("td", class_="rechts hauptlink")
            if mv_cell:
                market_value_str = mv_cell.get_text(strip=True)
                
                if market_value_str and market_value_str != "-" and "€" in market_value_str:
                    value = parse_market_value(market_value_str)
                    if value is not None:
                        return value
        
        return None
        
    except Exception as e:
        logging.error(f"Error searching for {player_name}: {e}")
        return None


def process_player(args):
    """
    Process a single player (to be run in a thread)
    
    Args:
        args: Tuple of (idx, player_name, club_name, age, output_csv)
    
    Returns:
        Tuple of (idx, player_name, market_value, success)
    """
    idx, player_name, club_name, age, output_csv = args
    
    driver = None
    try:
        # Each thread gets its own driver
        driver = setup_driver()
        
        logging.info(f"[{idx + 1}] Processing: {player_name}")
        
        # Get market value
        market_value = get_market_value(
            driver=driver,
            player_name=player_name,
            club_name=club_name,
            age=age
        )
        
        if market_value is not None:
            logging.info(f"[{idx + 1}] ✓ {player_name}: €{market_value}m")
            return (idx, player_name, market_value, True)
        else:
            logging.warning(f"[{idx + 1}] ✗ {player_name}: Not found")
            return (idx, player_name, None, False)
            
    except Exception as e:
        logging.error(f"[{idx + 1}] Error processing {player_name}: {e}")
        return (idx, player_name, None, False)
        
    finally:
        if driver:
            driver.quit()


def update_csv_market_values(input_csv: str, output_csv: str = None, resume: bool = True, num_threads: int = 4):
    """
    Read CSV file, search for each player's market value using multiple threads
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, will overwrite input file)
        resume: Whether to resume from checkpoint if exists
        num_threads: Number of concurrent threads (default: 4, recommended: 3-8)
    """
    if output_csv is None:
        output_csv = input_csv
    
    # Read CSV
    logging.info(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check if required columns exist
    if 'player_name' not in df.columns:
        raise ValueError("CSV must contain 'player_name' column")
    
    # Add market_value column if it doesn't exist
    if 'market_value' not in df.columns:
        df['market_value'] = None
    
    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {'processed_indices': set(), 'updated_count': 0, 'failed_count': 0}
    processed_indices = set(checkpoint.get('processed_indices', []))
    updated_count = checkpoint['updated_count']
    failed_count = checkpoint['failed_count']
    
    total_players = len(df)
    remaining_players = total_players - len(processed_indices)
    
    if len(processed_indices) > 0:
        logging.info(f"Resuming: {len(processed_indices)} players already processed, {remaining_players} remaining")
    
    logging.info(f"Starting multi-threaded scraping with {num_threads} threads...")
    
    # Prepare tasks for unprocessed players
    tasks = []
    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue
            
        player_name = row['player_name']
        current_club = row.get('current_club', None)
        age = row.get('age', None)
        
        # Convert age to int
        if pd.notna(age):
            try:
                age = int(float(age))
            except (ValueError, TypeError):
                age = None
        else:
            age = None
        
        tasks.append((idx, player_name, current_club if pd.notna(current_club) else None, age, output_csv))
    
    logging.info(f"Processing {len(tasks)} players with {num_threads} threads...")
    
    try:
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_player, task): task for task in tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    idx, player_name, market_value, success = future.result()
                    
                    # Update DataFrame
                    if market_value is not None:
                        df.at[idx, 'market_value'] = market_value
                        updated_count += 1
                    else:
                        failed_count += 1
                    
                    # Mark as processed
                    processed_indices.add(idx)
                    
                    # Save to CSV immediately (thread-safe)
                    with file_lock:
                        df.to_csv(output_csv, index=False)
                    
                    # Update checkpoint
                    checkpoint['processed_indices'] = processed_indices
                    checkpoint['updated_count'] = updated_count
                    checkpoint['failed_count'] = failed_count
                    save_checkpoint(checkpoint)
                    
                    # Log progress
                    progress = (len(processed_indices) / total_players) * 100
                    if len(processed_indices) % 10 == 0:
                        logging.info(f"Progress: {progress:.1f}% ({len(processed_indices)}/{total_players})")
                        
                except Exception as e:
                    logging.error(f"Error processing future: {e}")
        
        # Final save
        with file_lock:
            df.to_csv(output_csv, index=False)
        
        # Delete checkpoint after successful completion
        delete_checkpoint()
        
        # Summary
        logging.info("\n" + "="*60)
        logging.info("SUMMARY")
        logging.info("="*60)
        logging.info(f"Total players processed: {total_players}")
        logging.info(f"Successfully updated: {updated_count}")
        logging.info(f"Failed to update: {failed_count}")
        logging.info(f"Success rate: {(updated_count/total_players)*100:.2f}%")
        logging.info(f"Output saved to: {output_csv}")
        logging.info("="*60)
        
    except KeyboardInterrupt:
        logging.info("\n\nProcess interrupted by user")
        logging.info(f"Progress saved. Run again to resume from checkpoint")
        
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        logging.info(f"Progress saved. Run again to resume from checkpoint")
        raise


if __name__ == "__main__":
    input_file = r"D:\CNTT\IntroDS\Final_Project\Introduction-to-Data-Science-CSC14119-Final\source\Data Collection\football_first_half_dataset.csv"
    output_file = "players_updated.csv"
    
    try:
        # num_threads: Recommended 3-8 threads
        # Too many threads may get blocked by the server
        update_csv_market_values(input_file, output_file, resume=True, num_threads=5)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise