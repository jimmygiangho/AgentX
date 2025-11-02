"""
AXFI Pipeline Runner
CLI for running full pipeline or manual triggers
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def trigger_daily_scan(config: dict):
    """Trigger daily scan manually"""
    logger.info("Triggering daily scan...")
    
    # TODO: Implement full daily scan
    # - Load S&P 500 symbols
    # - Fetch data for all symbols
    # - Run strategy scanning
    # - AI ranking
    # - Generate top 10 recommendations
    # - Save reports
    
    print("Daily scan triggered (stub implementation)")
    print("Will implement full scan pipeline in Step 6")


def run_full_pipeline(config: dict):
    """Run complete pipeline"""
    logger.info("Running full AXFI pipeline...")
    
    # TODO: Orchestrate all agents
    print("Full pipeline triggered (stub implementation)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AXFI Pipeline Runner")
    parser.add_argument("--trigger-scan", action="store_true", help="Trigger daily scan")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.trigger_scan:
        trigger_daily_scan(config)
    elif args.full_pipeline:
        run_full_pipeline(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

