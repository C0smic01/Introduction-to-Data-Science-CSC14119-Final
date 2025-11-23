from utils.imports import *
from utils.config import MAX_DELAY, MIN_DELAY


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
