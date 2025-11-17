import random
import time

from config import MAX_DELAY, MIN_DELAY


def random_delay(a: float = MIN_DELAY, b: float = MAX_DELAY) -> None:
    """Add random delay to avoid rate limiting"""
    time.sleep(random.uniform(a, b))
