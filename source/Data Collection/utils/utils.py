from utils.imports import *
from utils.config import BASE_SCHEMA, MAX_DELAY, MIN_DELAY


def random_delay(a: float = MIN_DELAY, b: float = MAX_DELAY) -> None:
    """
    Pause execution for a random amount of time between `a` and `b` seconds.

    This is useful to mimic human-like behavior in scripts and avoid
    triggering rate limits when making repeated requests to APIs or websites.

    Args:
        a (float): Minimum number of seconds to sleep. Defaults to MIN_DELAY.
        b (float): Maximum number of seconds to sleep. Defaults to MAX_DELAY.
    """
    time.sleep(random.uniform(a, b))


def combine_csv(folder_path: str, output_path: str):
    """
    Combine all CSV files in a specified folder into a single CSV file.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        output_path (str): Path to the output CSV file.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print("Cannot find any CSV files in the specified folder.")
        return

    df_list = [pd.read_csv(file) for file in csv_files]

    # Combine all DataFrames into a single DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"Successfully merged {len(csv_files)} files into: {output_path}.")


def save_to_csv(self, filename: str) -> None:
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

    logging.info(f"Saved {len(self.players)} players to {filename}")


def save_to_json(self, filename: str) -> None:
    """Save all scraped players to JSON"""
    if not self.players:
        logging.warning("No players to save")
        return

    clean_players = [
        {k: p.get(k, BASE_SCHEMA[k]) for k in BASE_SCHEMA.keys()} for p in self.players
    ]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(clean_players, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved {len(self.players)} players to {filename}")


def generate_player_id(
    player_name: str, dob: Optional[str] = None, nationality: Optional[str] = None
) -> str:
    """
    Generate a unique player ID based on player name, date of birth, and nationality.
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


def clean_number(val, allow_float: bool = True):
    """Convert string or number to numeric value, handling commas, %, empty strings."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace(",", "").replace("%", "")
        if s == "":
            return None
        if allow_float and (
            "." in s or s.replace(".", "", 1).replace("-", "", 1).isdigit()
        ):
            return float(s)
        return int(float(s))
    except Exception:
        return None


def find_table_in_comments(
    soup: BeautifulSoup, needle: Optional[str] = None, id_contains: Optional[str] = None
):
    """Find a table that is hidden inside HTML comments."""
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if needle and needle not in c:
            continue
        try:
            s2 = BeautifulSoup(c, "html.parser")
            t = (
                s2.find("table", id=lambda x: x and id_contains in x)
                if id_contains
                else s2.find("table")
            )
            if t:
                return t
        except Exception:
            continue
    return None
