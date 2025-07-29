import os
import json
import logging
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, Asset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Fetch Alpaca API credentials
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL")

if not all([APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL]):
    raise ValueError("Missing Alpaca API credentials in .env file")

# Initialize Alpaca API client
api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)

def fetch_iex_symbols(save_to: str = "iex_symbols.json") -> list[str]:
    """
    Fetch tradable symbols available on IEX-only Alpaca feed and save to a JSON file.
    """
    logger.info("Fetching list of tradable assets from Alpaca...")
    all_assets = api.list_assets(status='active')

    # No need to filter by 'exchange' since all available assets use IEX routing
    iex_symbols = [
        asset.symbol
        for asset in all_assets
        if asset.tradable
    ]

    logger.info(f"Found {len(iex_symbols)} tradable symbols.")
    
    with open(save_to, "w") as f:
        json.dump(iex_symbols, f, indent=2)
        logger.info(f"Saved {len(iex_symbols)} symbols to {save_to}")

    return iex_symbols

if __name__ == "__main__":
    symbols = fetch_iex_symbols()
