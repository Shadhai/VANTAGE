"""Logging utilities for VANTAGE pipeline."""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"vantage_run_{timestamp}.log"
        self.error_file = self.log_dir / f"vantage_errors_{timestamp}.log"
        
        self.errors = []
        self.warnings = []
        self.start_time = None
    
    def start(self):
        """Start pipeline timer."""
        self.start_time = time.time()
        self._log("=" * 60)
        self._log("VANTAGE Pipeline Started")
        self._log(f"Time: {datetime.now().isoformat()}")
        self._log("=" * 60)
    
    def info(self, message: str):
        """Log info message."""
        self._log(f"[INFO] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        self.warnings.append(message)
        self._log(f"[WARN] {message}")
    
    def error(self, message: str, fatal: bool = False):
        """Log error message."""
        self.errors.append(message)
        self._log(f"[ERROR] {message}")
        self._log_error(message)
        
        if fatal:
            self._log("[FATAL] Pipeline stopped due to error")
            self.summary()
            sys.exit(1)
    
    def section(self, title: str):
        """Log section header."""
        self._log("")
        self._log(f"--- {title} ---")
    
    def summary(self):
        """Print pipeline summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self._log("")
        self._log("=" * 60)
        self._log("PIPELINE SUMMARY")
        self._log("=" * 60)
        self._log(f"Total time: {elapsed:.1f} seconds")
        self._log(f"Errors: {len(self.errors)}")
        self._log(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            self._log("\nErrors:")
            for e in self.errors:
                self._log(f"  - {e}")
        
        if self.warnings:
            self._log("\nWarnings:")
            for w in self.warnings:
                self._log(f"  - {w}")
    
    def _log(self, message: str):
        """Write to console and log file."""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    def _log_error(self, message: str):
        """Write to error file."""
        with open(self.error_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")