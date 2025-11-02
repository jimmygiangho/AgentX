"""
Test suite for AXFI scaffold
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import main_scaffold


def test_main_returns_ok():
    """Test that main() returns 'AXFI scaffold OK'"""
    assert main_scaffold.main() == "AXFI scaffold OK", "main() should return 'AXFI scaffold OK'"


if __name__ == "__main__":
    test_main_returns_ok()
    print("✅ All scaffold tests passed")

