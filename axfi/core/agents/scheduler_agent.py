"""
Scheduler Agent - Orchestrates daily pipeline
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    logging.warning("APScheduler not available")

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SchedulerAgent(BaseAgent):
    """Agent responsible for scheduling and orchestration"""
    
    def __init__(self, config: dict, **kwargs):
        """Initialize scheduler agent"""
        super().__init__(config, **kwargs)
        
        if APSCHEDULER_AVAILABLE:
            # Use system timezone or UTC
            try:
                from tzlocal import get_localzone
                tz = str(get_localzone())
            except:
                tz = "UTC"
            self.scheduler = BackgroundScheduler(timezone=tz)
            self.scheduler.start()
            logger.info(f"APScheduler started with timezone {tz}")
        else:
            self.scheduler = None
            logger.warning("APScheduler not available")
    
    def schedule_daily_scan(self):
        """Schedule daily scan at configured time"""
        if not self.scheduler:
            logger.error("APScheduler not available")
            return
        
        # Get scan time from config
        scan_config = self.config.get("scan", {})
        run_time = scan_config.get("run_time", "23:00")
        
        # Parse hour and minute
        hour, minute = map(int, run_time.split(":"))
        
        # Schedule job
        self.scheduler.add_job(
            self.daily_scan_job,
            trigger=CronTrigger(hour=hour, minute=minute),
            id='daily_scan_job',
            name='Daily S&P 500 Scan',
            replace_existing=True
        )
        
        logger.info(f"Scheduled daily scan at {run_time}")
    
    def daily_scan_job(self):
        """
        Main job function for daily S&P 500 scan.
        Runs at scheduled time.
        """
        logger.info("="*80)
        logger.info("DAILY SCAN JOB TRIGGERED")
        logger.info("="*80)
        
        try:
            # 1. Run scanner agent
            from core.agents.scanner_agent import ScannerAgent
            from core.reports import ReportsGenerator
            
            scanner = ScannerAgent(self.config)
            scan_result = scanner.run(symbols=None, top_n=10)
            
            if scan_result['status'] != 'success' or not scan_result['recommendations']:
                logger.warning("No recommendations generated")
                return
            
            # 2. Generate reports
            reports_gen = ReportsGenerator(
                reports_dir=self.config.get('report', {}).get('out_dir', './reports')
            )
            
            # Save CSV
            csv_path = reports_gen.generate_timestamp_filename("scan_top10", "csv")
            df = pd.DataFrame(scan_result['recommendations'])
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved scan_top10 CSV: {csv_path}")
            
            # Save HTML
            html_path = reports_gen.generate_timestamp_filename("scan_top10", "html")
            html_content = self._generate_top10_html(scan_result['recommendations'])
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Saved scan_top10 HTML: {html_path}")
            
            logger.info(f"Daily scan completed: {len(scan_result['recommendations'])} top recommendations")
            
        except Exception as e:
            logger.error(f"Error in daily scan job: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _generate_top10_html(self, recommendations: List[Dict]) -> str:
        """Generate HTML report for top 10 recommendations"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>AXFI Daily Scan - Top 10 Recommendations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background: #34495e; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f0f0f0; }
        .score { font-weight: bold; }
        .confidence { background: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; }
        .explanation { font-size: 0.9em; color: #555; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AXFI Daily Scan Results</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p>Top 10 Trade Recommendations</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Score</th>
                <th>Entry</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Confidence</th>
                <th>Allocation %</th>
                <th>Explanation</th>
            </tr>
        </thead>
        <tbody>"""
        
        for idx, rec in enumerate(recommendations, 1):
            html += f"""
            <tr>
                <td><strong>{idx}</strong></td>
                <td><strong>{rec['symbol']}</strong></td>
                <td>{rec['strategy']}</td>
                <td class="score">{rec['score']:.4f}</td>
                <td>${rec['entry']:.2f}</td>
                <td>${rec['stop_loss']:.2f}</td>
                <td>${rec['take_profit']:.2f}</td>
                <td><span class="confidence">{rec['confidence']}%</span></td>
                <td>{rec['suggested_pct']}%</td>
                <td class="explanation">{rec['explanation']}</td>
            </tr>"""
        
        html += """
        </tbody>
    </table>
</body>
</html>"""
        return html
    
    def trigger_scan_now(self):
        """Manually trigger the daily scan immediately"""
        logger.info("Manual scan trigger requested")
        self.daily_scan_job()
    
    def run(self, **kwargs) -> dict:
        """Run scheduler setup"""
        if self.scheduler:
            self.schedule_daily_scan()
            jobs = self.scheduler.get_jobs()
            logger.info(f"Scheduler running with {len(jobs)} jobs")
            return {
                "status": "running",
                "jobs": [{"id": j.id, "name": j.name} for j in jobs]
            }
        else:
            return {
                "status": "disabled",
                "message": "APScheduler not available"
            }
    
    def shutdown(self):
        """Shutdown scheduler"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down")


if __name__ == "__main__":
    # Test scheduler agent
    import yaml
    
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    agent = SchedulerAgent(config)
    result = agent.run()
    print(f"Scheduler result: {result}")
    
    # Keep running for testing
    import time
    print("\nScheduler running. Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        agent.shutdown()
