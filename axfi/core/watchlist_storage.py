"""
Watchlist storage module for managing user watchlists
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Get the project root directory (axfi folder)
PROJECT_ROOT = Path(__file__).parent.parent
WATCHLIST_FILE = PROJECT_ROOT / "db" / "watchlists.json"


def load_watchlists() -> Dict:
    """
    Load watchlists from storage file.
    
    Returns:
        Dictionary with watchlists data
    """
    try:
        if WATCHLIST_FILE.exists():
            with open(WATCHLIST_FILE, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded watchlists from storage")
            return data
    except Exception as e:
        logger.warning(f"Error loading watchlists: {e}")
    
    # Return default structure if file doesn't exist
    return {
        "watchlists": [],
        "default_lists": {
            "S&P 500": {
                "name": "S&P 500",
                "symbols": [],
                "readonly": True,
                "created_at": datetime.now().isoformat()
            },
            "DOW 100": {
                "name": "DOW 100",
                "symbols": [],
                "readonly": True,
                "created_at": datetime.now().isoformat()
            }
        },
        "updated_at": datetime.now().isoformat()
    }


def save_watchlists(data: Dict):
    """
    Save watchlists to storage file.
    
    Args:
        data: Dictionary with watchlists data
    """
    try:
        WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = datetime.now().isoformat()
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Saved watchlists to storage")
    except Exception as e:
        logger.error(f"Error saving watchlists: {e}")
        raise


def get_watchlist(name: str) -> Optional[Dict]:
    """
    Get a specific watchlist by name.
    
    Args:
        name: Watchlist name
        
    Returns:
        Watchlist dictionary or None if not found
    """
    data = load_watchlists()
    
    # Check default lists
    if name in data.get("default_lists", {}):
        return data["default_lists"][name]
    
    # Check custom watchlists
    for wl in data.get("watchlists", []):
        if wl["name"] == name:
            return wl
    
    return None


def create_watchlist(name: str, symbols: Optional[List[str]] = None) -> bool:
    """
    Create a new custom watchlist.
    
    Args:
        name: Watchlist name
        symbols: Optional list of symbols to add
        
    Returns:
        True if created successfully
    """
    data = load_watchlists()
    
    # Check if watchlist already exists
    if get_watchlist(name):
        logger.warning(f"Watchlist '{name}' already exists")
        return False
    
    # Create new watchlist
    new_watchlist = {
        "name": name,
        "symbols": symbols or [],
        "readonly": False,
        "created_at": datetime.now().isoformat()
    }
    
    data.setdefault("watchlists", []).append(new_watchlist)
    save_watchlists(data)
    logger.info(f"Created watchlist '{name}'")
    return True


def delete_watchlist(name: str) -> bool:
    """
    Delete a custom watchlist.
    
    Args:
        name: Watchlist name
        
    Returns:
        True if deleted successfully
    """
    data = load_watchlists()
    
    # Check if it's a default list (can't delete)
    if name in data.get("default_lists", {}):
        logger.warning(f"Cannot delete default watchlist '{name}'")
        return False
    
    # Remove from custom watchlists
    data["watchlists"] = [wl for wl in data.get("watchlists", []) if wl["name"] != name]
    save_watchlists(data)
    logger.info(f"Deleted watchlist '{name}'")
    return True


def add_symbol_to_watchlist(name: str, symbol: str) -> bool:
    """
    Add a symbol to a watchlist.
    
    Args:
        name: Watchlist name
        symbol: Stock symbol to add
        
    Returns:
        True if added successfully
    """
    data = load_watchlists()
    symbol = symbol.upper().strip()
    
    # Check default lists
    if name in data.get("default_lists", {}):
        wl = data["default_lists"][name]
        if symbol not in wl["symbols"]:
            wl["symbols"].append(symbol)
            save_watchlists(data)
            logger.info(f"Added {symbol} to default watchlist '{name}'")
            return True
        return False
    
    # Check custom watchlists
    for wl in data.get("watchlists", []):
        if wl["name"] == name:
            if symbol not in wl["symbols"]:
                wl["symbols"].append(symbol)
                save_watchlists(data)
                logger.info(f"Added {symbol} to watchlist '{name}'")
                return True
            return False
    
    logger.warning(f"Watchlist '{name}' not found")
    return False


def remove_symbol_from_watchlist(name: str, symbol: str) -> bool:
    """
    Remove a symbol from a watchlist.
    
    Args:
        name: Watchlist name
        symbol: Stock symbol to remove
        
    Returns:
        True if removed successfully
    """
    data = load_watchlists()
    symbol = symbol.upper().strip()
    
    # Check default lists
    if name in data.get("default_lists", {}):
        wl = data["default_lists"][name]
        if symbol in wl["symbols"]:
            wl["symbols"].remove(symbol)
            save_watchlists(data)
            logger.info(f"Removed {symbol} from default watchlist '{name}'")
            return True
        return False
    
    # Check custom watchlists
    for wl in data.get("watchlists", []):
        if wl["name"] == name:
            if symbol in wl["symbols"]:
                wl["symbols"].remove(symbol)
                save_watchlists(data)
                logger.info(f"Removed {symbol} from watchlist '{name}'")
                return True
            return False
    
    logger.warning(f"Watchlist '{name}' not found")
    return False


def initialize_default_lists():
    """
    Initialize default watchlists with S&P 500 and DOW 100 symbols.
    """
    import yaml
    
    data = load_watchlists()
    
    # Load S&P 500 symbols from JSON
    try:
        sp500_file = PROJECT_ROOT / "data" / "sp500_symbols.json"
        if sp500_file.exists():
            with open(sp500_file, 'r') as f:
                sp500_data = json.load(f)
                sp500_symbols = sp500_data.get("symbols", [])
            logger.info(f"Loaded {len(sp500_symbols)} S&P 500 symbols from JSON")
        else:
            # Fallback to config.yaml
            config_file = PROJECT_ROOT / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    sp500_symbols = config.get("sp500_symbols", [])
                logger.info(f"Loaded {len(sp500_symbols)} S&P 500 symbols from config.yaml")
            else:
                sp500_symbols = []
                logger.warning("S&P 500 symbols file not found")
    except Exception as e:
        logger.error(f"Error loading S&P 500 symbols: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sp500_symbols = []
    
    # Load DOW 100 symbols
    try:
        dow_file = PROJECT_ROOT / "data" / "dow100_symbols.json"
        if dow_file.exists():
            with open(dow_file, 'r') as f:
                dow_data = json.load(f)
                dow_symbols = dow_data.get("symbols", [])
            logger.info(f"Loaded {len(dow_symbols)} DOW 100 symbols from JSON")
        else:
            dow_symbols = []
            logger.warning("DOW 100 symbols file not found")
    except Exception as e:
        logger.error(f"Error loading DOW 100 symbols: {e}")
        import traceback
        logger.error(traceback.format_exc())
        dow_symbols = []
    
    # Update default lists
    if "default_lists" not in data:
        data["default_lists"] = {}
    
    # Always update S&P 500 list (will populate if empty, or keep existing if populated)
    if "S&P 500" not in data["default_lists"]:
        data["default_lists"]["S&P 500"] = {
            "name": "S&P 500",
            "symbols": sp500_symbols,
            "readonly": True,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Created S&P 500 default list with {len(sp500_symbols)} symbols")
    else:
        # Update symbols if list exists but is empty
        if not data["default_lists"]["S&P 500"]["symbols"] and sp500_symbols:
            data["default_lists"]["S&P 500"]["symbols"] = sp500_symbols
            logger.info(f"Populated empty S&P 500 list with {len(sp500_symbols)} symbols")
        else:
            logger.info(f"S&P 500 list already has {len(data['default_lists']['S&P 500']['symbols'])} symbols")
    
    # Always update DOW 100 list (will populate if empty, or keep existing if populated)
    if "DOW 100" not in data["default_lists"]:
        data["default_lists"]["DOW 100"] = {
            "name": "DOW 100",
            "symbols": dow_symbols,
            "readonly": True,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Created DOW 100 default list with {len(dow_symbols)} symbols")
    else:
        # Update symbols if list exists but is empty
        if not data["default_lists"]["DOW 100"]["symbols"] and dow_symbols:
            data["default_lists"]["DOW 100"]["symbols"] = dow_symbols
            logger.info(f"Populated empty DOW 100 list with {len(dow_symbols)} symbols")
        else:
            logger.info(f"DOW 100 list already has {len(data['default_lists']['DOW 100']['symbols'])} symbols")
    
    save_watchlists(data)
    logger.info(f"Initialized default watchlists: S&P 500 ({len(data['default_lists']['S&P 500']['symbols'])} symbols), DOW 100 ({len(data['default_lists']['DOW 100']['symbols'])} symbols)")

