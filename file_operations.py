"""
File operations module for handling file system operations.
"""

import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileOperations:
    """Handles file system operations."""
    
    def __init__(self, work_dir: Path):
        """Initialize file operations."""
        self.work_dir = work_dir
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_local_directories(self):
        """Clean local directories for safety."""
        logger.info("Cleaning local work directories...")
        
        if self.work_dir.exists():
            for item in self.work_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        logger.info("Local directories cleaned")

