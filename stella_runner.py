"""
Stella route calculation runner for Linux environment.
Fetches tasks with waiting_for_route_calculation status, downloads stella frames zip,
calculates route using Stella VSLAM, and uploads results to API.
"""

import json
import time
import logging
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional, List

from api_client import APIClient
from file_operations import FileOperations
from route_calculation import RouteCalculation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StellaRunner:
    """Handles Stella route calculation for tasks waiting for route calculation."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize Stella runner with configuration."""
        # Load configuration from JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        api_domain = self.config.get('api_domain')
        api_key = self.config.get('api_key')
        stella_exec = self.config.get('stella_executable')
        stella_config_path = self.config.get('stella_config_path')
        stella_vocab_path = self.config.get('stella_vocab_path')
        stella_results_path = self.config.get('stella_results_path')
        
        if not api_domain or not api_key:
            raise ValueError("api_domain and api_key must be set in config.json")
        if not all([stella_exec, stella_config_path, stella_vocab_path]):
            raise ValueError("stella_executable, stella_config_path, and stella_vocab_path must be set in config.json")
        
        self.polling_interval = self.config.get('polling_interval', 15)
        self.work_dir = Path(self.config.get('local_work_dir', './work'))
        self.candidates_per_second = self.config.get('candidates_per_second', 12)
        
        # Initialize modules
        self.api_client = APIClient(api_domain, api_key)
        self.update_status_text = self.api_client.update_status_text
        self.file_ops = FileOperations(self.work_dir)
        self.route_calculation = RouteCalculation(
            self.work_dir,
            stella_exec,
            stella_config_path,
            stella_vocab_path,
            stella_results_path or '',
            self.candidates_per_second
        )
    
    def clean_local_directories(self):
        """Clean local directories for safety."""
        if self.api_client.current_task_id:
            self.update_status_text("Siivotaan paikalliset hakemistot...")
        self.file_ops.clean_local_directories()
        
        # Also clean stella_results directory if it exists
        stella_results_path = self.config.get('stella_results_path')
        if stella_results_path:
            try:
                stella_results_dir = Path(stella_results_path)
                if not stella_results_dir.is_absolute():
                    stella_results_dir = self.work_dir / stella_results_dir
                
                if stella_results_dir.exists():
                    logger.info(f"Cleaning stella_results directory: {stella_results_dir}")
                    shutil.rmtree(stella_results_dir)
                    logger.info("Stella results directory cleaned")
            except Exception as e:
                logger.warning(f"Failed to clean stella_results directory: {e}")
    
    def fetch_next_stella_task(self) -> Optional[Dict]:
        """Fetch next task with waiting_for_route_calculation status."""
        logger.info("Fetching next Stella route calculation task...")
        url = "/api/v1/process-recording-task/get-next-task"
        return self.api_client._api_request('GET', url, params={"task_type": "route_calculation"})
    def download_file(self, file_uuid: str, output_path: Path) -> bool:
        """Download file from API using file UUID."""
        try:
            logger.info(f"Fetching file data for UUID {file_uuid}...")
            file_data = self.api_client._api_request('GET', f'/api/v1/file/{file_uuid}/')
            
            if not file_data:
                logger.error("Failed to fetch file data from API")
                return False
            
            file_url = file_data.get('url')
            if not file_url:
                logger.error("Missing URL in file data")
                return False
            
            logger.info(f"Downloading file from {file_url}...")
            response = requests.get(file_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"File downloaded to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """Extract zip file and return list of extracted frame paths."""
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting {zip_path} to {extract_dir}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find all image files in extracted directory
            frame_paths = sorted(extract_dir.rglob("*.jpg")) + sorted(extract_dir.rglob("*.png"))
            logger.info(f"Extracted {len(frame_paths)} frames from zip")
            return frame_paths
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}")
            return []
    
    def process_task(self, task: Dict) -> bool:
        """Process a single Stella route calculation task."""
        self.api_client.current_task_id = task.get('uuid')
        self.api_client.current_video_recording_id = task.get('video_recording')
        logger.info(f"Processing Stella task {self.api_client.current_task_id}")
        self.update_status_text("Aloitetaan Stella reitin laskenta...")
        
        try:
            # Step 1: Clean directories
            self.clean_local_directories()
            
            # Step 2: Get stella_frames_zip_uuid from task details
            details = task.get('details', {})
            if not details:
                raise Exception("Task missing details field")
            
            stella_zip_uuid = details.get('stella_frames_zip_uuid')
            if not stella_zip_uuid:
                raise Exception("Task details missing stella_frames_zip_uuid")
            
            logger.info(f"Stella frames zip UUID: {stella_zip_uuid}")
            
            # Step 3: Download zip file
            self.update_status_text("Downloading Stella frame zip file...")
            zip_path = self.work_dir / "stella_frames.zip"
            if not self.download_file(stella_zip_uuid, zip_path):
                raise Exception("Failed to download stella frames zip file")
            
            if not zip_path.exists() or zip_path.stat().st_size == 0:
                raise Exception(f"Downloaded zip file is missing or empty: {zip_path}")
            
            # Step 4: Extract zip file
            self.update_status_text("Puretaan Stella frame zip-tiedosto...")
            frames_dir = self.work_dir / "stella_frames"
            frame_paths = self.extract_zip(zip_path, frames_dir)
            
            if not frame_paths:
                raise Exception("No frames extracted from zip file")
            
            logger.info(f"Extracted {len(frame_paths)} frames for route calculation")
            self.update_status_text(f"Purettu {len(frame_paths)} framea zip-tiedostosta")
            
            # Step 5: Calculate route using Stella
            route_data = self.route_calculation.calculate_route(
                frame_paths,
                self.update_status_text
            )
            
            if not route_data:
                raise Exception("Stella route calculation failed")
            
            # Step 6: Get video-recording to find processed video
            video_recording_id = self.api_client.current_video_recording_id
            if not video_recording_id:
                raise Exception("Missing video_recording_id in task")
            
            video_recording = self.api_client.fetch_video_recording(video_recording_id)
            if not video_recording:
                raise Exception("Failed to fetch video-recording data")
            
            # Find processed stitched video
            videos = video_recording.get('videos', [])
            processed_video = None
            for video in videos:
                if video.get('category') == 'video_360_field_raw':
                    processed_video = video
                    break
            
            if not processed_video:
                # Fallback: use first video if processed video not found
                if videos:
                    processed_video = videos[0]
                    logger.warning("Processed video not found, using first available video")
                else:
                    raise Exception("No video found in video-recording")
            
            processed_video_id = processed_video.get('uuid')
            if not processed_video_id:
                raise Exception("Processed video missing UUID")
            
            # Step 7: Update video with raw_path
            raw_path = route_data.get('raw_path')
            if raw_path:
                self.update_status_text("Tallennetaan reitti API:in...")
                if not self.api_client.update_task_route(processed_video_id, raw_path):
                    logger.warning("Failed to update video with raw_path")
                else:
                    logger.info("Route data saved to video")
            else:
                logger.warning("Route data returned without raw_path information")
            
            # Step 8: Clean local directories
            self.update_status_text("Siivotaan paikalliset hakemistot...")
            self.clean_local_directories()
            
            # Step 9: Report completion
            self.update_status_text("Stella reitin laskenta valmis")
            self.api_client.report_task_completion(success=True)
            logger.info(f"Stella task {self.api_client.current_task_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Stella task processing failed: {e}")
            self.update_status_text(f"Error: {str(e)}")
            self.api_client.report_task_completion(success=False, error=str(e))
            self.clean_local_directories()
            return False
    
    def run(self):
        """Main loop: poll for Stella tasks and process them."""
        logger.info("Starting Stella route calculation runner...")
        logger.info(f"Polling interval: {self.polling_interval} seconds")
        
        while True:
            try:
                task = self.fetch_next_stella_task()
                task_id = task.get('uuid') if task else None
                
                if task and task_id:
                    logger.info(f"Found Stella task: {task_id}")
                    self.process_task(task)
                else:
                    logger.debug("No Stella tasks available, waiting...")
                
                time.sleep(self.polling_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.polling_interval)


if __name__ == "__main__":
    runner = StellaRunner()
    runner.run()

