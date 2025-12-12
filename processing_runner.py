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
        # WSL configuration - force WSL mode without Docker
        use_wsl = True  # Always use WSL
        use_docker = False  # Never use Docker
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
        self.frame_upload_parallelism = self.config.get('frame_upload_parallelism', 8)
        
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
        
        # Track current task status (valid values: "pending", "in_progress", "completed", "failed")
        self.current_task_status = "in_progress"
        self.last_status_text = None  # Track last sent status to avoid duplicate updates
        
        # Create wrapper for status updates that automatically uses current task ID
        # Only sends update if status text has changed (to reduce API load)
        def update_status_text(status_text: str, status: Optional[str] = None, force: bool = False):
            """Update task status text using current task ID.
            
            Only sends update if status text has changed from last update (to reduce API load).
            
            Args:
                status_text: Status message text (goes to 'result' field)
                status: Optional status value ("pending", "in_progress", "completed", "failed")
                        If not provided, uses "in_progress" as default
                force: If True, always send update even if text hasn't changed
            """
            if self.api_client.current_task_id:
                # Only update if text changed or forced
                if force or status_text != self.last_status_text:
                    # Clean status text: remove null characters and other control characters
                    cleaned_text = status_text.replace('\u0000', '').replace('\x00', '').replace('\r', ' ').replace('\n', ' ').strip()
                    # Limit length to avoid issues
                    if len(cleaned_text) > 500:
                        cleaned_text = cleaned_text[:497] + "..."
                    
                    # Use provided status or default to "in_progress"
                    task_status = status or "in_progress"
                    self.media_server_api_client.update_task_status(
                        self.api_client.current_task_id,
                        task_status,
                        result=cleaned_text
                    )
                    self.last_status_text = status_text
        self.update_status_text = update_status_text
        self.file_ops = FileOperations(self.work_dir)
        self.video_processing = VideoProcessing(self.work_dir, mediasdk_executable, stitch_config)
        # Configure frame processing with source and target FPS for direct frame extraction
        source_fps = 29.95  # Default source FPS
        target_fps = 10.0   # Target FPS (every 3rd frame from 29.95 fps)
        self.frame_processing = FrameProcessing(
            self.work_dir, 
            self.candidates_per_second, 
            self.low_res_fps, 
            self.blur_settings,
            source_fps=source_fps,
            target_fps=target_fps
        )
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
        
        task_id = task.get('task_id')
        video_recording_id = task.get('video_recording')
        self.api_client.current_task_id = task_id
        self.api_client.current_video_recording_id = video_recording_id

        if not video_recording_id:
            logger.error("Reset task missing video_recording ID")
            return False

        # Reset last status text when starting new task
        self.last_status_text = None
        self.update_status_text("Reset: starting video-recording cleanup", force=True)
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
                self.api_client.set_video_recording_status(
                    self.api_client.current_video_recording_id,
                    status='completed'
                )
        else:
            self.update_status_text("Reset: cleanup failed")
            if not test_mode:
                self.api_client.set_video_recording_status(
                    self.api_client.current_video_recording_id,
                    status='failure'
                )
        return success
    
    def process_task(self, task: Dict) -> bool:
        """Process a single video task through all steps."""
        # Update API client credentials from task
        self._update_api_client_from_task(task)
        
        self.api_client.current_task_id = task.get('task_id')
        self.api_client.test_mode = False
        self.api_client.current_video_recording_id = task.get('details', {}).get('video_recording')
        logger.info(f"Processing task {self.api_client.current_task_id}")
        
        # Reset last status text when starting new task
        self.last_status_text = None
        self.update_status_text("Starting video processing task...", status="in_progress", force=True)
        
        # Update video-recording status to "processing"
        if self.api_client.current_video_recording_id:
            self.api_client.set_video_recording_status(
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
            
            # Step 4: Stitch videos to H265 MP4 using MediaSDK
            stitched_path = self.work_dir / "stitched_video.mp4"
            if not self.video_processing.stitch_videos(
                front_path,
                back_path,
                stitched_path,
                self.update_status_text
            ):
                raise Exception("Failed to stitch videos")
            
            # Step 5: Get video duration from stitched video
            video_duration = self.video_processing.get_video_duration(stitched_path)
            if not video_duration:
                # Try to get from front video as fallback
                video_duration = self.video_processing.get_video_duration(front_path)
                if not video_duration:
                    logger.warning("Could not determine video duration, defaulting to 10 minutes")
                    video_duration = 600.0  # 10 minutes
            
            # Step 6: Extract frames from stitched video using FFmpeg
            # First extract at candidates_per_second fps for frame selection (1 fps for API)
            # create_and_select_frames will extract frames and select best ones
            selected_high, selected_low = self.frame_processing.create_and_select_frames(
                stitched_path,
                self.update_status_text
            )
            if not selected_high or not selected_low:
                raise Exception(f"Failed to select frames (got {len(selected_high)} high and {len(selected_low)} low frames)")
            
            # Step 7: Extract all frames at 10 fps for Stella route calculation
            # Extract high-res frames (max resolution) and low-res frames (3840x1920) at 10 fps
            stella_high_dir = self.work_dir / "stella_high_frames"
            stella_low_dir = self.work_dir / "stella_low_frames"
            stella_high_frames, stella_low_frames = self.frame_processing.extract_frames_high_and_low(
                stitched_path,
                stella_high_dir,
                stella_low_dir,
                fps=10.0  # Extract at 10 fps for Stella route calculation
            )
            if not stella_low_frames:
                raise Exception(f"Failed to extract Stella frames (got {len(stella_low_frames)} low frames)")
            
            # Step 8: Store stitched video to API
            stitched_video_id = None
            if project_id:
                video_type = self.config.get('processed_video_category', 'video_insta360_processed_stitched')
                stitched_video_id = self.video_processing.store_processed_video(
                    stitched_path,  # Upload stitched video binary
                    project_id,
                    video_recording,
                    self.api_client,
                    video_type=video_type,
                    video_duration=video_duration
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
            
            # Update video duration to video_recording
            if video_duration and video_recording_id:
                self.api_client.update_video_recording_duration(video_recording_id, video_duration)
            
            # Step 9: Optional blur (only for selected frames that go to API)
            final_high, final_low = self.frame_processing.blur_frames_optional(
                selected_high,
                selected_low,
                self.blur_people,
                self.update_status_text
            )
            
            # Step 10: Upload selected frames to cloud (1 fps)
            layer_id = video_recording.get('layer')
            frame_objects = self.frame_processing.upload_frames_to_cloud(
                final_high + final_low,
                project_id,
                stitched_video_id,
                self.api_client,
                self.update_status_text,
                layer_id=layer_id,
                max_workers=self.frame_upload_parallelism
            )
            
            # Step 11: Prepare all low-res frames (10 fps) for Stella route calculation
            # Use all low-res frames extracted at 10 fps (not just selected ones)
            route_frames = self.frame_processing.get_route_frames_from_low_res(
                stella_low_frames,  # All 10 fps low-res frames for Stella
                self.update_status_text
            )
            logger.info(f"Using {len(route_frames)} frames (10 fps) for Stella route calculation")
            
            # Step 12: Calculate route using Stella VSLAM (via WSL if configured)
            route_data = self.route_calculation.calculate_route(
                route_frames,
                stitched_path,  # Use actual stitched video path
                self.update_status_text
            )
            if route_data:
                raw_path = route_data.get('raw_path')
                route_duration = route_data.get('duration')
                
                # Use route duration if available, otherwise use video duration
                if route_duration:
                    if abs(route_duration - video_duration) > 5.0:  # More than 5 seconds difference
                        logger.warning(f"Route duration ({route_duration:.2f}s) differs significantly from video duration ({video_duration:.2f}s)")
                    # Update video duration if route provides it
                    if video_recording_id:
                        self.api_client.update_video_recording_duration(video_recording_id, route_duration)
                
                if raw_path:
                    # Update raw_path to the stitched video
                    if self.processed_video_id:
                        self.api_client.update_task_route(self.processed_video_id, raw_path)
                    else:
                        logger.warning("Cannot update raw_path: processed video ID not available")
                    
                    # Update video frames with camera_layer_position from raw_path
                    layer_id = video_recording.get('layer')
                    if layer_id and self.processed_video_id:
                        self.update_status_text("Updating video frames with camera positions from route data...")
                        # Extract starting_position and ending_position from video_recording if available
                        # API uses 'starting_position' and 'ending_position' (not 'start_position' and 'end_position')
                        start_position = video_recording.get('starting_position')
                        end_position = video_recording.get('ending_position')
                        updated_count = self.api_client.update_video_frames_from_raw_path(
                            self.processed_video_id,
                            raw_path,
                            layer_id,
                            self.update_status_text,
                            start_position=start_position,
                            end_position=end_position
                        )
                        logger.info(f"Updated {updated_count} video frames with camera positions")
                    else:
                        if not layer_id:
                            logger.warning("Cannot update video frames: missing layer_id")
                        if not self.processed_video_id:
                            logger.warning("Cannot update video frames: missing processed_video_id")
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
            self.update_status_text("Task complete, starting new task polling...", status="completed")
            
            # Update video-recording status to "ready_to_view" on success
            if self.api_client.current_video_recording_id:
                self.api_client.set_video_recording_status(
                    self.api_client.current_video_recording_id,
                    status='ready_to_view'
                )
            
            logger.info(f"Task {self.api_client.current_task_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            # Clean error message: remove null characters and other control characters
            error_msg = str(e)
            cleaned_error = error_msg.replace('\u0000', '').replace('\x00', '').replace('\r', ' ').replace('\n', ' ').strip()
            if len(cleaned_error) > 200:
                cleaned_error = cleaned_error[:197] + "..."
            self.update_status_text(f"Error: {cleaned_error}", status="failed", force=True)
            
            # Update video-recording status to "failure" on error
            if self.api_client.current_video_recording_id:
                self.api_client.set_video_recording_status(
                    self.api_client.current_video_recording_id,
                    status='failure'
                )
            
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
                    if success:
                        logger.info("Test mode: Ensuring completion status is set...")
                        self.api_client.set_video_recording_status(
                            self.api_client.current_video_recording_id,
                            status='completed'
                        )
                        logger.info("Test mode: Task processing completed successfully")
                    else:
                        logger.error("Test mode: Task processing failed")
                        self.api_client.set_video_recording_status(
                            self.api_client.current_video_recording_id,
                            status='failure'
                        )
                except Exception as e:
                    logger.error(f"Test mode: Exception during processing: {e}")
                    self.api_client.set_video_recording_status(
                        self.api_client.current_video_recording_id,
                        status='failure'
                    )
                logger.info("Test mode complete (reset + processing done)")
            else:
                logger.warning("No processing task found in test mode")
            return

        # Normal mode: continue with regular loop
        effective_reset = reset
        while True:
            try:
                task = self.media_server_api_client.fetch_next_task(reset=effective_reset)
                task_id = task.get('task_id') if task else None
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
