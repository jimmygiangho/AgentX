"""
Cache for storing last scan results to persist across dashboard loads
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

CACHE_FILE = Path("db/last_scan_results.json")


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
        
        logger.info(f"Saved {len(recommendations)} recommendations to cache")
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
            logger.info(f"Loaded {data.get('total_recommendations', 0)} recommendations from cache")
            return data
    except Exception as e:
        logger.warning(f"Error loading scan results: {e}")
    
    return None

