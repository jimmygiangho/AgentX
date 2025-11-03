"""
Quick test script to verify watchlist API routes are accessible
"""
import requests
import sys

def test_api(base_url="http://127.0.0.1:8081"):
    """Test the watchlist API endpoints"""
    try:
        # Test health endpoint
        print(f"Testing {base_url}/health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
        
        # Test watchlists endpoint
        print(f"\nTesting {base_url}/api/watchlists...")
        response = requests.get(f"{base_url}/api/watchlists", timeout=5)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {data}")
            print(f"  Default lists: {list(data.get('default_lists', {}).keys())}")
            print(f"  Custom watchlists: {len(data.get('watchlists', []))}")
        else:
            print(f"  Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {base_url}")
        print("Make sure the server is running!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()

