# Logging utilities module
# Configures logging for the entire trading system # logging.py
"""
Logging Utilities Module
Configures logging for the entire trading system.
"""

import logging
import os
from datetime import datetime

def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_file_prefix: str = "trading_system"
):
    """
    Set up global logging configuration.
    Logs to both console and file (rotates daily).
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"{log_file_prefix}_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    log_format = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )
    logging.getLogger().info("Logging is configured. Log file: %s", log_file)

# Optional: Call setup_logging() on import, or require explicit call in main.py
if __name__ == "__main__":
    setup_logging()
    logging.info("This is an info log.")
    logging.warning("This is a warning.")
    logging.error("This is an error.")
