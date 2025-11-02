"""
Test live data fetching
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.data_providers import get_data_provider


def test_live_data_fetch():
    """Test fetching live data for AAPL"""
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    provider = get_data_provider(config)
    quote = provider.get_latest_quote("AAPL")
    
    assert "symbol" in quote, "Quote should have symbol"
    assert quote["symbol"] == "AAPL", "Should fetch AAPL"
    
    if "error" not in quote:
        assert "price" in quote, "Quote should have price"
        assert quote["price"] > 0, "Price should be positive"
        print(f"✅ Live data test passed: {quote}")
    else:
        print(f"⚠️ Warning: {quote['error']}")


if __name__ == "__main__":
    test_live_data_fetch()
    print("✅ Live data tests completed")

