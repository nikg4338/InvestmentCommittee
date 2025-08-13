#!/usr/bin/env python3
"""
Market Hours Scheduler
=====================

Manages autonomous trading schedule during market hours.
"""

import schedule
import time
import logging
import pytz
from datetime import datetime, timedelta
from typing import Callable, List
import threading

logger = logging.getLogger(__name__)

class MarketScheduler:
    """
    Manages scheduling for autonomous trading during market hours.
    """
    
    def __init__(self, timezone: str = "US/Eastern"):
        self.timezone = pytz.timezone(timezone)
        self.is_running = False
        self.scheduled_jobs = []
        
    def schedule_market_hours_job(self, 
                                 job_func: Callable,
                                 interval_minutes: int = 5,
                                 job_name: str = "market_job"):
        """
        Schedule a job to run during market hours only.
        
        Args:
            job_func: Function to execute
            interval_minutes: How often to run (in minutes)
            job_name: Name for logging
        """
        def market_hours_wrapper():
            if self._is_market_open():
                try:
                    logger.info(f"Executing {job_name}")
                    job_func()
                except Exception as e:
                    logger.error(f"Error in {job_name}: {e}")
            else:
                logger.debug(f"Market closed - skipping {job_name}")
                
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(market_hours_wrapper)
        self.scheduled_jobs.append(job_name)
        logger.info(f"Scheduled {job_name} every {interval_minutes} minutes during market hours")
        
    def schedule_daily_job(self, 
                          job_func: Callable,
                          time_str: str,
                          job_name: str = "daily_job"):
        """
        Schedule a daily job at specific time.
        
        Args:
            job_func: Function to execute
            time_str: Time in "HH:MM" format
            job_name: Name for logging
        """
        def daily_wrapper():
            try:
                logger.info(f"Executing daily job: {job_name}")
                job_func()
            except Exception as e:
                logger.error(f"Error in daily job {job_name}: {e}")
                
        schedule.every().day.at(time_str).do(daily_wrapper)
        self.scheduled_jobs.append(f"{job_name}_daily")
        logger.info(f"Scheduled daily job {job_name} at {time_str}")
        
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(self.timezone)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if it's within trading hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    def start_scheduler(self):
        """Start the scheduler in a separate thread."""
        self.is_running = True
        
        def run_scheduler():
            logger.info("📅 Market scheduler started")
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        return scheduler_thread
        
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("📅 Market scheduler stopped")
