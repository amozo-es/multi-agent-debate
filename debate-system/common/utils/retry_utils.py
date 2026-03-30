"""Retry utilities for handling API rate limits."""

import subprocess
import time
import random
from typing import List


def run_with_retries(cmd: List[str], max_retries: int = 2, base_delay: float = 15) -> bool:
    """
    Execute a subprocess command with exponential backoff retries.

    Args:
        cmd: Command to execute as list of strings
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries

    Returns:
        True if command succeeded, False otherwise
    """
    attempt = 0
    while True:
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            attempt += 1
            if attempt > max_retries:
                print(f"??  Error permanente (returncode={e.returncode}). Sin mas reintentos.")
                return False
            # Backoff exponencial con jitter para aliviar el rate limit
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 3)
            print(f"??  Fallo (rc={e.returncode}). Reintentando en {delay:.1f}s (intento {attempt}/{max_retries})...")
            time.sleep(delay)
        except Exception as e:
            print(f"??  Unexpected error: {e}. Sin reintentos para este caso.")
            return False


def add_delay_between_requests(base_delay: float, jitter: float = 4) -> float:
    """
    Calculate and add delay between API requests.

    Args:
        base_delay: Base delay in seconds
        jitter: Maximum random jitter to add

    Returns:
        Total delay time
    """
    delay = base_delay + random.uniform(0, jitter)
    print(f"⏱️  Waiting {delay:.1f}s before next request...")
    time.sleep(delay)
    return delay