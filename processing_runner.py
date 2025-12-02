"""
Video processing script for Insta360 material.
Fetches tasks from API, processes videos, and reports results.
"""

import json
import time
import logging
import shutil
import psutil
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from api_client import APIClient, MediaServerAPIClient
from file_operations import FileOperations
from video_processing import VideoProcessing
from frame_processing import FrameProcessing
from route_calculation import RouteCalculation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing class."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize processor with configuration."""
        # Load configuration from JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # API configuration (defaults from config, can be overridden by tasks)
        default_api_domain = self.config.get('api_domain')
        default_api_key = self.config.get('api_key')
        media_server_api_key = self.config.get('media_server_api_key')
        media_server_api_domain = self.config.get('media_server_api_domain')
        worker_id = self.config.get('worker_id')
        
        # Stella configuration
        stella_exec = self.config.get('stella_executable')
        stella_config_path = self.config.get('stella_config_path')
        stella_vocab_path = self.config.get('stella_vocab_path')
        stella_results_path = self.config.get('stella_results_path')
        use_wsl = self.config.get('stella_use_wsl', False)
        use_docker = self.config.get('stella_use_docker', False)
        docker_image = self.config.get('stella_docker_image', 'stella_vslam-socket')
        docker_data_mount = self.config.get('stella_docker_data_mount', '/data')
        
        # Stitching configuration
        mediasdk_executable = self.config.get('mediasdk_executable')
        stitch_config = self.config.get('stitch_config', {})
        
        # Blur configuration
        self.blur_people = False  # Can be set from task data if needed
        self.blur_settings = self.config.get('blur_settings', {})
        
        # Frame configuration
        self.frames_per_second = self.config['frames_per_second']
        self.low_res_fps = self.config['low_res_frames_per_second']
        self.route_fps = self.config['route_calculation_fps']
        self.candidates_per_second = self.config.get('candidates_per_second', 12)
        
        # General processing configuration
        self.polling_interval = self.config['polling_interval']
        self.work_dir = Path(self.config['local_work_dir'])
        self.grace_period_days = self.config['video_deletion_grace_period_days']
        
        # Validation
        if not mediasdk_executable:
            raise ValueError("mediasdk_executable must be set in config.json")
        if not media_server_api_key:
            raise ValueError("media_server_api_key must be set in config.json")
        if not worker_id:
            raise ValueError("worker_id must be set in config.json")
        if not media_server_api_domain:
            raise ValueError("media_server_api_domain must be set in config.json")
        
        # Initialize modules (API client can be updated per task)
        self.api_client = APIClient(default_api_domain, default_api_key)
        self.media_server_api_client = MediaServerAPIClient(media_server_api_domain, media_server_api_key, worker_id)
        self.update_status_text = self.media_server_api_client.update_task_status  # direct alias
        self.file_ops = FileOperations(self.work_dir)
        self.video_processing = VideoProcessing(self.work_dir, mediasdk_executable, stitch_config)
        self.frame_processing = FrameProcessing(self.work_dir, self.candidates_per_second, self.low_res_fps, self.blur_settings)
        self.route_calculation = RouteCalculation(
            self.work_dir,
            stella_exec or '',
            stella_config_path or '',
            stella_vocab_path or '',
            stella_results_path or '',
            self.candidates_per_second,
            use_wsl=use_wsl,
            use_docker=use_docker,
            docker_image=docker_image,
            docker_data_mount=docker_data_mount
        )
        self.processed_video_id: Optional[str] = None
        self.test_task_uuid: Optional[str] = self.config.get('test_task_uuid')
    
    def clean_local_directories(self):
        """Clean local directories for safety."""
        if self.api_client.current_task_id:
            self.update_status_text("Cleaning local directories...")
        self.file_ops.clean_local_directories()
        
        # Also clean stella_results directory if it exists
        stella_results_path = self.config.get('stella_results_path')
        if stella_results_path:
            try:
                stella_results_dir = Path(stella_results_path)
                # Handle relative paths (e.g., ./work/stella_results)
                if not stella_results_dir.is_absolute():
                    stella_results_dir = self.work_dir / stella_results_dir
                
                if stella_results_dir.exists():
                    logger.info(f"Cleaning stella_results directory: {stella_results_dir}")
                    shutil.rmtree(stella_results_dir)
                    logger.info("Stella results directory cleaned")
            except Exception as e:
                logger.warning(f"Failed to clean stella_results directory: {e}")
    
    def _update_api_client_from_task(self, task: Dict):
        """Update API client credentials from task data if available."""
        api_domain = task.get('domain')
        api_key = task.get('token')
        if api_domain and api_key:
            self.api_client.update_credentials(api_domain, api_key)
            logger.info(f"Updated API client from task (domain: {api_domain})")
        elif not self.api_client.api_domain or not self.api_client.api_key:
            raise ValueError("API credentials missing: neither task nor config provides domain and token")
    
    def handle_reset_task(self, task: Dict, test_mode: bool = False) -> bool:
        """Process a reset task by cleaning API data."""
        # Update API client credentials from task
        self._update_api_client_from_task(task)
        
        task_id = task.get('uuid')
        video_recording_id = task.get('video_recording')
        self.api_client.current_task_id = task_id
        self.api_client.current_video_recording_id = video_recording_id

        if not video_recording_id:
            logger.error("Reset task missing video_recording ID")
            return False

        self.update_status_text("Reset: starting video-recording cleanup")
        preserve_categories = self.config.get('fallback_video_categories', [
            'video_insta360_raw_front',
            'video_insta360_raw_back'
        ])
        reset_status = self.config.get('test_video_recording_reset_status', 'created')
        success = self.api_client.cleanup_video_recording_data(
            video_recording_id,
            self.update_status_text,
            preserve_categories=preserve_categories,
            test_mode=test_mode,
            test_reset_status=reset_status
        )
        if success:
            self.update_status_text("Reset: cleanup complete")
            if not test_mode:
                self.api_client.report_task_completion(success=True)
        else:
            self.update_status_text("Reset: cleanup failed")
            if not test_mode:
                self.api_client.report_task_completion(success=False, error="Reset cleanup failed")
        return success
    
    def process_task(self, task: Dict) -> bool:
        """Process a single video task through all steps."""
        # Update API client credentials from task
        self._update_api_client_from_task(task)
        
        self.api_client.current_task_id = task.get('uuid')
        self.api_client.test_mode = task.get('is_test', False)
        self.api_client.current_video_recording_id = task.get('video_recording')
        logger.info(f"Processing task {self.api_client.current_task_id}")
        self.update_status_text("Starting video processing task...")
        
        # Update video-recording status to "processing"
        if self.api_client.current_video_recording_id:
            self.api_client.reset_video_recording_status(
                self.api_client.current_video_recording_id,
                status='processing'
            )
        
        try:
            # Step 2: Clean directories
            self.clean_local_directories()
            
            # Step 3: Download videos
            video_recording_id = self.api_client.current_video_recording_id
            if not video_recording_id:
                raise Exception("Missing video_recording_id in task")
            
            video_recording = self.api_client.fetch_video_recording(video_recording_id)
            if not video_recording:
                raise Exception("Failed to fetch video-recording data")
            
            front_path, back_path = self.video_processing.download_videos(
                video_recording,
                self.api_client,
                self.update_status_text
            )
            if not front_path or not back_path:
                raise Exception("Failed to download videos")

            project_id = video_recording.get('project')
            if not project_id:
                raise Exception("Video recording missing project reference")
            if video_recording.get('blur_people') is not None:
                self.blur_people = bool(video_recording.get('blur_people'))
            
            # Step 4: Stitch videos
            stitched_path = self.work_dir / "stitched_output.mp4"
            if not self.video_processing.stitch_videos(
                front_path,
                back_path,
                stitched_path,
                self.update_status_text
            ):
                raise Exception("Failed to stitch videos")
            
            stitched_video_id = None
            if project_id:
                video_type = self.config.get('processed_video_category', 'video_insta360_processed_stitched')
                stitched_video_id = self.video_processing.store_processed_video(
                    stitched_path,
                    project_id,
                    video_recording,
                    self.api_client,
                    video_type=video_type
                )
            if not stitched_video_id:
                fallback_categories = self.config.get('fallback_video_categories', [
                    'video_insta360_raw_front',
                    'video_insta360_raw_back'
                ])
                stitched_video_id = self.video_processing.select_fallback_video_id(
                    video_recording,
                    fallback_categories=fallback_categories
                )
            if not stitched_video_id:
                raise Exception("Failed to determine target video_id for frames")
            self.processed_video_id = stitched_video_id
            
            # Step 5: Create and select frames (12fps high and low res, select sharpest)
            if not stitched_path.exists():
                raise Exception(f"Stitched video not found at {stitched_path}")
            if stitched_path.stat().st_size == 0:
                raise Exception(f"Stitched video is empty at {stitched_path}")
            
            selected_high, selected_low = self.frame_processing.create_and_select_frames(
                stitched_path,
                self.update_status_text
            )
            if not selected_high or not selected_low:
                raise Exception(f"Failed to create and select frames (got {len(selected_high)} high and {len(selected_low)} low frames)")
            
            # Step 6: Optional blur
            final_high, final_low = self.frame_processing.blur_frames_optional(
                selected_high,
                selected_low,
                self.blur_people,
                self.update_status_text
            )
            
            # Step 7: Upload frames to cloud
            layer_id = video_recording.get('layer')
            frame_objects = self.frame_processing.upload_frames_to_cloud(
                final_high + final_low,
                project_id,
                stitched_video_id,
                self.api_client,
                self.update_status_text,
                layer_id=layer_id
            )
            
            # Step 8: Create 12fps frames for Stella route calculation
            route_frames = self.frame_processing.create_stella_frames(
                stitched_path,
                self.update_status_text
            )
            if not route_frames:
                raise Exception("Failed to create Stella frames for route calculation")
            
            # Step 9: Calculate route using Stella VSLAM (via WSL if configured)
            route_data = self.route_calculation.calculate_route(
                route_frames,
                self.update_status_text
            )
            if route_data:
                raw_path = route_data.get('raw_path')
                if raw_path:
                    # Update raw_path to the stitched video
                    if self.processed_video_id:
                        self.api_client.update_task_route(self.processed_video_id, raw_path)
                    else:
                        logger.warning("Cannot update raw_path: processed video ID not available")
                else:
                    logger.warning("Route data returned without raw_path information")
            
            # Step 10: Mark videos for deletion
            if self.api_client.current_task_id and not self.api_client.test_mode:
                self.update_status_text(f"Marking front and back videos for deletion in {self.grace_period_days} days (grace period)...")
                result = self.api_client.mark_videos_for_deletion(self.grace_period_days)
                if result:
                    self.update_status_text("Videos marked for deletion")
            
            # Step 11: Clean local directories
            self.update_status_text("Cleaning local directories...")
            self.clean_local_directories()
            
            # Report success
            self.update_status_text("Task complete, starting new task polling...")
            self.api_client.report_task_completion(success=True)
            logger.info(f"Task {self.api_client.current_task_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.update_status_text(f"Error: {str(e)}")
            self.api_client.report_task_completion(success=False, error=str(e))
            self.clean_local_directories()
            return False
    
    def run(self, reset: bool = False, test_mode: bool = False):
        """Main loop: poll for tasks and process them."""
        logger.info("Starting video processor...")
        if test_mode:
            logger.info("Running in TEST mode (reset + normal processing)")
            # Enable test mode in API client for detailed logging
            self.api_client.test_mode = True
        elif reset:
            logger.info("Running in RESET mode")
        logger.info(f"Polling interval: {self.polling_interval} seconds")

        if test_mode:
            if not self.test_task_uuid:
                logger.error("Test mode requested but config.json missing test_task_uuid")
                return
            if self.media_server_api_client.update_task_status(self.test_task_uuid, 'pending'):
                logger.info(f"Test task {self.test_task_uuid} reset to pending")
            else:
                logger.warning(f"Failed to reset test task {self.test_task_uuid} to pending")

            # Test mode: first do reset, then normal processing
            logger.info("Test mode: Step 1 - Resetting and cleaning data...")
            task = self.media_server_api_client.fetch_next_task(reset=True)
            task_id = task.get('uuid') if task else None
            if task and task_id:
                logger.info(f"Found reset task: {task_id}")
                self.handle_reset_task(task, test_mode=test_mode)
                logger.info("Reset complete, waiting briefly before fetching task for processing...")
                time.sleep(2)  # Brief pause to ensure API state is updated
            else:
                logger.warning("No reset task found in test mode")
            
            # Step 2: Reset task status to pending before fetching for processing
            logger.info("Test mode: Resetting task status to pending before processing...")
            if self.media_server_api_client.update_task_status(self.test_task_uuid, 'pending'):
                logger.info(f"Test task {self.test_task_uuid} reset to pending for processing")
            else:
                logger.warning(f"Failed to reset test task {self.test_task_uuid} to pending")
            
            # Step 3: Fetch task again and do normal processing
            logger.info("Test mode: Step 2 - Processing task normally...")
            task = self.media_server_api_client.fetch_next_task(reset=False)
            task_id = task.get('uuid') if task else None
            if task and task_id:
                logger.info(f"Found processing task: {task_id}")
                # Note: Backend may set status to 'processing' when task is fetched
                # This is expected behavior
                try:
                    success = self.process_task(task)
                    # process_task already calls report_task_completion, but ensure it's done
                    if success:
                        logger.info("Test mode: Ensuring completion status is set...")
                        self.api_client.report_task_completion(success=True)
                        logger.info("Test mode: Task processing completed successfully")
                    else:
                        logger.error("Test mode: Task processing failed")
                        self.api_client.report_task_completion(success=False, error="Processing failed in test mode")
                except Exception as e:
                    logger.error(f"Test mode: Exception during processing: {e}")
                    self.api_client.report_task_completion(success=False, error=str(e))
                logger.info("Test mode complete (reset + processing done)")
            else:
                logger.warning("No processing task found in test mode")
            return

        # Normal mode: continue with regular loop
        effective_reset = reset
        while True:
            try:
                task = self.media_server_api_client.fetch_next_task(reset=effective_reset)
                task_id = task.get('uuid') if task else None
                if task and task_id:
                    logger.info(f"Found task: {task_id}")
                    if effective_reset:
                        self.handle_reset_task(task, test_mode=False)
                    else:
                        self.process_task(task)
                else:
                    logger.debug("No tasks available, waiting...")

                if test_mode:
                    break

                time.sleep(self.polling_interval)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                if test_mode:
                    break
                time.sleep(self.polling_interval)


if __name__ == "__main__":
    import argparse
    
    # Kill other instances of this script before starting
    current_pid = os.getpid()
    script_name = os.path.basename(__file__)
    
    killed_count = 0
    try:
        if os.name == 'nt':  # Windows
            # Find all Python processes running processing_runner.py
            result = subprocess.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'],
                capture_output=True, text=True, timeout=10
            )
            
            for line in result.stdout.splitlines():
                if script_name in line and str(current_pid) not in line:
                    # Extract PID (first number in line)
                    parts = line.split()
                    for part in parts:
                        try:
                            pid = int(part)
                            if pid != current_pid:
                                logger.info(f"Killing existing instance (PID: {pid})")
                                subprocess.run(
                                    ['taskkill', '/F', '/PID', str(pid)],
                                    capture_output=True,
                                    stderr=subprocess.DEVNULL,
                                    timeout=5
                                )
                                killed_count += 1
                                break
                        except ValueError:
                            continue
        else:  # Linux/Unix
            # Use psutil if available, otherwise use pgrep
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any(script_name in arg for arg in cmdline):
                                pid = proc.info['pid']
                                if pid != current_pid:
                                    logger.info(f"Killing existing instance (PID: {pid})")
                                    proc.kill()
                                    killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            except ImportError:
                # Fallback to pgrep if psutil not available
                try:
                    result = subprocess.run(
                        ['pgrep', '-f', script_name],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    for pid_str in result.stdout.strip().splitlines():
                        try:
                            pid = int(pid_str)
                            if pid != current_pid:
                                logger.info(f"Killing existing instance (PID: {pid})")
                                subprocess.run(['kill', '-9', str(pid)], timeout=5)
                                killed_count += 1
                        except ValueError:
                            pass
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        
        if killed_count > 0:
            logger.info(f"Killed {killed_count} existing instance(s)")
            time.sleep(1)  # Give processes time to die
    except Exception as e:
        logger.warning(f"Failed to check for existing instances: {e}")

    parser = argparse.ArgumentParser(description="Insta360 video processing runner")
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Run in reset mode (cleanup API objects instead of processing)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run single-iteration test (reset=true, uses config.test_task_uuid)'
    )
    args = parser.parse_args()

    processor = VideoProcessor()
    processor.run(reset=args.reset, test_mode=args.test)
