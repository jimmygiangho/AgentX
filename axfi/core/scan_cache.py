"""
Cache for storing last scan results to persist across dashboard loads
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Use absolute path based on project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_FILE = PROJECT_ROOT / "db" / "last_scan_results.json"


def save_scan_results(recommendations: List[Dict], timestamp: Optional[str] = None):
    """
    Save scan results to cache file.
    
    Args:
        recommendations: List of recommendation dictionaries
        timestamp: Optional timestamp string
    """
    try:
        cache_data = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }
        
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
            f.flush()  # Ensure data is written to buffer
            import os
            os.fsync(f.fileno())  # Force write to disk immediately
        
        logger.info(f"Saved {len(recommendations)} recommendations to cache with timestamp: {cache_data.get('timestamp')} at {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving scan results: {e}")


def load_scan_results() -> Optional[Dict]:
    """
    Load last scan results from cache.
    
    Returns:
        Dictionary with scan results or None if not found
    """
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            timestamp = data.get('timestamp', 'Not found')
            logger.info(f"Loaded {data.get('total_recommendations', 0)} recommendations from cache. Timestamp: {timestamp}")
            return data
        else:
            logger.warning(f"Cache file does not exist: {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error loading scan results from {CACHE_FILE}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return None

