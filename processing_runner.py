"""
Video processing script for Insta360 material.
Fetches tasks from API, processes videos, and reports results.
"""

import json
import time
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from api_client import APIClient
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
        
        api_domain = self.config.get('api_domain')
        api_key = self.config.get('api_key')
        mediasdk_executable = self.config.get('mediasdk_executable')
        stella_exec = self.config.get('stella_executable')
        stella_config_path = self.config.get('stella_config_path')
        stella_vocab_path = self.config.get('stella_vocab_path')
        stella_results_path = self.config.get('stella_results_path')
        self.enable_stella = self.config.get('enable_stella', False)
        
        if not api_domain or not api_key:
            raise ValueError("api_domain and api_key must be set in config.json")
        if not mediasdk_executable:
            raise ValueError("mediasdk_executable must be set in config.json")
        
        self.polling_interval = self.config['polling_interval']
        self.work_dir = Path(self.config['local_work_dir'])
        self.frames_per_second = self.config['frames_per_second']
        self.low_res_fps = self.config['low_res_frames_per_second']
        self.route_fps = self.config['route_calculation_fps']
        self.grace_period_days = self.config['video_deletion_grace_period_days']
        self.candidates_per_second = self.config.get('candidates_per_second', 12)
        self.stella_frames_category = self.config.get('stella_frames_category', 'general_zip')
        
        self.blur_people = False  # Can be set from task data if needed
        self.blur_settings = self.config.get('blur_settings', {})
        
        # Initialize modules
        self.api_client = APIClient(api_domain, api_key)
        self.update_status_text = self.api_client.update_status_text  # direct alias
        self.file_ops = FileOperations(self.work_dir)
        self.video_processing = VideoProcessing(self.work_dir, mediasdk_executable)
        self.frame_processing = FrameProcessing(self.work_dir, self.candidates_per_second, self.blur_settings)
        self.route_calculation = RouteCalculation(
            self.work_dir,
            stella_exec or '',
            stella_config_path or '',
            stella_vocab_path or '',
            stella_results_path or '',
            self.candidates_per_second
        )
        self.processed_video_id: Optional[str] = None
        self.test_task_uuid: Optional[str] = self.config.get('test_task_uuid')
    
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
                # Handle relative paths (e.g., ./work/stella_results)
                if not stella_results_dir.is_absolute():
                    stella_results_dir = self.work_dir / stella_results_dir
                
                if stella_results_dir.exists():
                    logger.info(f"Cleaning stella_results directory: {stella_results_dir}")
                    shutil.rmtree(stella_results_dir)
                    logger.info("Stella results directory cleaned")
            except Exception as e:
                logger.warning(f"Failed to clean stella_results directory: {e}")
    
    def download_videos(self, task: Dict) -> tuple[Optional[Path], Optional[Path]]:
        """Download front and back videos from video-recording data."""
        # Fetch video-recording data from API
        video_recording_id = self.api_client.current_video_recording_id
        if not video_recording_id:
            logger.error("Missing video_recording_id in task")
            self.update_status_text("Virhe: video_recording_id puuttuu")
            return None, None
        
        video_recording = self.api_client.fetch_video_recording(video_recording_id)
        if not video_recording:
            logger.error("Failed to fetch video-recording data")
            self.update_status_text("Virhe: Video-recording datan haku epäonnistui")
            return None, None
        
        return self.video_processing.download_videos(
            video_recording,
            self.api_client,
            self.update_status_text
        )
    
    def upload_frames_to_cloud(self, frame_paths: list[Path], project_id: Optional[str] = None, video_id: Optional[str] = None, layer_id: Optional[str] = None) -> list[Dict]:
        """Save frames to cloud and create frame objects."""
        if not project_id:
            # Try to get project_id from video_recording if available
            if self.api_client.current_video_recording_id:
                video_recording = self.api_client.fetch_video_recording(self.api_client.current_video_recording_id)
                if video_recording:
                    project_id = video_recording.get('project')
                    if not layer_id:
                        layer_id = video_recording.get('layer')
        
        if not project_id:
            logger.error("No project_id available for frame upload")
            self.update_status_text("Virhe: project_id puuttuu")
            return []
        
        return self.frame_processing.upload_frames_to_cloud(
            frame_paths,
            project_id,
            video_id,
            self.api_client,
            self.update_status_text,
            layer_id=layer_id
        )
    
    def get_route_frames_from_low_res(self, low_frames: list[Path]) -> list[Path]:
        """Use 12fps low res frames for route calculation."""
        return self.frame_processing.get_route_frames_from_low_res(
            low_frames,
            self.update_status_text
        )

    def package_stella_frames(self, frame_paths: list[Path]) -> Optional[Path]:
        """Package Stella frame inputs into a single ZIP archive."""
        if not frame_paths:
            logger.warning("No frames available to package for Stella routing")
            return None
        
        zip_path = self.work_dir / "stella_route_frames.zip"
        try:
            if zip_path.exists():
                zip_path.unlink()
            
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for frame in frame_paths:
                    if frame.exists():
                        zf.write(frame, arcname=frame.name)
                    else:
                        logger.warning(f"Frame missing during zip packaging: {frame}")
            
            logger.info(f"Packaged {len(frame_paths)} frames into {zip_path}")
            return zip_path
        except Exception as exc:
            logger.error(f"Failed to package Stella frames: {exc}")
            return None

    def upload_stella_frames_zip(self, zip_path: Path, project_id: str, video_recording: Optional[Dict]) -> Optional[str]:
        """Upload Stella route frames zip to API storage."""
        if not zip_path.exists():
            logger.warning(f"Stella frames zip not found at {zip_path}")
            return None
        
        try:
            with open(zip_path, 'rb') as f:
                zip_binary = f.read()
        except Exception as exc:
            logger.error(f"Failed to read Stella frames zip: {exc}")
            return None
        
        video_recording_id = video_recording.get('uuid') if video_recording else self.api_client.current_video_recording_id
        logger.info("Uploading Stella frames zip to API...")
        zip_uuid = self.api_client.store_zip_file(
            project_id=project_id,
            zip_binary=zip_binary,
            name=zip_path.name,
            category=self.stella_frames_category,
            video_recording_id=video_recording_id
        )
        if zip_uuid:
            logger.info(f"Stella frames zip stored with ID {zip_uuid}")
        else:
            logger.warning("Failed to store Stella frames zip")
        return zip_uuid

    def _extract_image_id(self, image_field) -> Optional[str]:
        """Helper to extract UUID from image reference."""
        if not image_field:
            return None
        if isinstance(image_field, dict):
            return image_field.get('uuid') or image_field.get('id')
        if isinstance(image_field, str):
            return image_field
        return None

    def _delete_frame_and_images(self, frame: Dict) -> bool:
        """Delete images linked to a frame, then delete the frame."""
        frame_id = frame.get('uuid')
        image_fields = [
            'high_res_image',
            'low_res_image',
            'blur_high_image',
            'blur_image'
        ]
        success = True
        for field in image_fields:
            image_id = self._extract_image_id(frame.get(field))
            if image_id:
                if not self.api_client.delete_image(image_id):
                    success = False
        if frame_id:
            if not self.api_client.delete_video_frame(frame_id):
                success = False
        return success

    def cleanup_video_recording_data(self, video_recording_id: str, test_mode: bool = False) -> bool:
        """Delete videos, frames, and images associated with a video-recording."""
        video_recording = self.api_client.fetch_video_recording(video_recording_id)
        if not video_recording:
            logger.error("Reset cleanup failed: video-recording not found")
            return False

        videos = video_recording.get('videos', [])
        targets: List[Tuple[str, bool]] = []
        # Categories to preserve (front and back raw videos)
        preserve_categories = [
            'video_insta360_raw_front',
            'video_insta360_raw_back'
        ]
        
        for video in videos:
            video_uuid = video.get('uuid')
            category = video.get('category')
            # Skip front and back raw videos - don't delete them
            if video_uuid and category not in preserve_categories:
                targets.append((video_uuid, True))
        # Frame entries may reference video_recording_id directly
        targets.append((video_recording_id, False))

        for video_id, delete_video_flag in targets:
            self.update_status_text(f"Reset: poistetaan framet videolle {video_id}")
            frames = self.api_client.fetch_video_frames(video_id)
            for frame in frames:
                self._delete_frame_and_images(frame)
            if delete_video_flag:
                self.update_status_text(f"Reset: poistetaan video {video_id}")
                self.api_client.delete_video(video_id)

        if test_mode:
            # In test mode, reset status instead of deleting to avoid cascade deletion of process-recording-task
            self.update_status_text("Reset: resetoidaan video-recording status")
            reset_status = self.config.get('test_video_recording_reset_status', 'created')
            self.api_client.reset_video_recording_status(video_recording_id, status=reset_status)
            logger.info(f"Reset cleanup finished for video-recording {video_recording_id} (status reset to {reset_status})")
        else:
            # In normal reset mode, delete the video-recording
            self.update_status_text("Reset: poistetaan video-recording")
            self.api_client.delete_video_recording(video_recording_id)
            logger.info(f"Reset cleanup finished for video-recording {video_recording_id}")
        return True

    def _select_fallback_video_id(self, video_recording: Optional[Dict]) -> Optional[str]:
        """Choose an existing video UUID to associate frames if processed video missing."""
        if not video_recording:
            return None
        videos = video_recording.get('videos', []) or []
        preferred_categories = self.config.get('fallback_video_categories', [
            'video_insta360_raw_front',
            'video_insta360_raw_back'
        ])
        for category in preferred_categories:
            for video in videos:
                if video.get('category') == category and video.get('uuid'):
                    return video.get('uuid')
        if videos:
            return videos[0].get('uuid')
        return None

    def store_processed_video(self, stitched_path: Path, project_id: str, video_recording: Optional[Dict]) -> Optional[str]:
        """Upload stitched video binary to API and return video UUID."""
        if not stitched_path.exists():
            logger.warning(f"Processed video not found at {stitched_path}")
            return None
        try:
            with open(stitched_path, 'rb') as f:
                video_binary = f.read()
            video_type = self.config.get('processed_video_category', 'video_insta360_processed_stitched')
            video_name = stitched_path.name
            video_recording_id = video_recording.get('uuid') if video_recording else self.api_client.current_video_recording_id
            video_id = self.api_client.store_video(
                project_id=project_id,
                video_recording_id=video_recording_id,
                video_type=video_type,
                video_size=len(video_binary),
                video_binary=video_binary,
                name=video_name
            )
            if video_id:
                logger.info(f"Stored processed stitched video with ID {video_id}")
            return video_id
        except Exception as exc:
            logger.error(f"Failed to store processed video: {exc}")
            return None

    def handle_reset_task(self, task: Dict, test_mode: bool = False) -> bool:
        """Process a reset task by cleaning API data."""
        task_id = task.get('uuid')
        video_recording_id = task.get('video_recording')
        self.api_client.current_task_id = task_id
        self.api_client.current_video_recording_id = video_recording_id

        if not video_recording_id:
            logger.error("Reset task missing video_recording ID")
            return False

        self.update_status_text("Reset: aloitetaan video-recordingin siivous")
        success = self.cleanup_video_recording_data(video_recording_id, test_mode=test_mode)
        if success:
            self.update_status_text("Reset: siivous valmis")
            if not test_mode:
                self.api_client.report_task_completion(success=True)
        else:
            self.update_status_text("Reset: siivous epäonnistui")
            if not test_mode:
                self.api_client.report_task_completion(success=False, error="Reset cleanup failed")
        return success
    
    def calculate_route(self, frame_paths: list[Path]) -> Optional[Dict]:
        """Calculate route using selected frames (Stella VSLAM) and update raw path."""
        return self.route_calculation.calculate_route(
            frame_paths,
            self.update_status_text
        )
    
    def process_task(self, task: Dict) -> bool:
        """Process a single video task through all steps."""
        self.api_client.current_task_id = task.get('uuid')
        self.api_client.test_mode = task.get('is_test', False)
        self.api_client.current_video_recording_id = task.get('video_recording')
        logger.info(f"Processing task {self.api_client.current_task_id}")
        self.update_status_text("Aloitetaan videokäsittelytehtävä...")
        
        try:
            # Step 2: Clean directories
            self.clean_local_directories()
            
            # Step 3: Download videos
            front_path, back_path = self.download_videos(task)
            if not front_path or not back_path:
                raise Exception("Failed to download videos")

            video_recording = self.api_client.fetch_video_recording(self.api_client.current_video_recording_id)
            if not video_recording:
                raise Exception("Failed to reload video recording data")
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
                stitched_video_id = self.store_processed_video(stitched_path, project_id, video_recording)
            if not stitched_video_id:
                stitched_video_id = self._select_fallback_video_id(video_recording)
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
            # Get project_id and layer from video_recording
            layer_id = video_recording.get('layer')
            frame_objects = self.upload_frames_to_cloud(
                final_high + final_low,
                project_id=project_id,
                video_id=stitched_video_id,
                layer_id=layer_id
            )
            
            # Step 8: Use 12fps low res frames for route calculation
            route_frames = self.get_route_frames_from_low_res(selected_low)
            stella_zip_uuid: Optional[str] = None
            
            # Step 9: Calculate route (optional, based on enable_stella config)
            if self.enable_stella:
                route_data = self.calculate_route(route_frames)
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
            else:
                logger.info("Stella route calculation is disabled, skipping route calculation step")
                self.update_status_text("Stella reitin laskenta on pois käytöstä, ohitetaan...")
                self.update_status_text("Paketoi Stella framet zip-arkistoon...")
                stella_zip_path = self.package_stella_frames(route_frames)
                if stella_zip_path and project_id:
                    self.update_status_text("Ladataan Stella frame zip API:in...")
                    stella_zip_uuid = self.upload_stella_frames_zip(stella_zip_path, project_id, video_recording)
                    if stella_zip_uuid:
                        self.update_status_text("Stella frame zip tallennettu pilveen")
                    else:
                        self.update_status_text("Varoitus: Stella frame zip tallennus epäonnistui")
                else:
                    logger.warning("Stella frame zip packaging skipped (missing frames or project)")
                    self.update_status_text("Varoitus: Stella frame zip pakkaus epäonnistui")
            
            # Step 10: Mark videos for deletion
            if self.api_client.current_task_id and not self.api_client.test_mode:
                self.update_status_text(f"Merkataan front ja back video tuhottavaksi {self.grace_period_days} päivän päästä (varoaika)...")
                result = self.api_client.mark_videos_for_deletion(self.grace_period_days)
                if result:
                    self.update_status_text("Videot merkitty poistettavaksi")
            
            # Step 11: Clean local directories
            self.update_status_text("Siivotaan paikalliset hakemistot...")
            self.clean_local_directories()
            
            # Report success or waiting status
            if self.enable_stella:
                self.update_status_text("Tehtävä valmis, aloitetaan uuden tehtävän pollaus...")
                self.api_client.report_task_completion(success=True)
                logger.info(f"Task {self.api_client.current_task_id} processed successfully")
            else:
                self.update_status_text("Tehtävä valmis (odottaa reitin laskentaa), aloitetaan uuden tehtävän pollaus...")
                details_payload = {}
                if stella_zip_uuid:
                    details_payload['stella_frames_zip_uuid'] = stella_zip_uuid
                self.api_client.report_task_completion(
                    success=True,
                    waiting_for_route=True,
                    details=details_payload or None
                )
                logger.info(f"Task {self.api_client.current_task_id} processed successfully (waiting for route calculation)")
            return True
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.update_status_text(f"Virhe: {str(e)}")
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
            if self.api_client.set_task_status(self.test_task_uuid, 'pending'):
                logger.info(f"Test task {self.test_task_uuid} reset to pending")
            else:
                logger.warning(f"Failed to reset test task {self.test_task_uuid} to pending")

            # Test mode: first do reset, then normal processing
            logger.info("Test mode: Step 1 - Resetting and cleaning data...")
            task = self.api_client.fetch_next_task(reset=True)
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
            if self.api_client.set_task_status(self.test_task_uuid, 'pending'):
                logger.info(f"Test task {self.test_task_uuid} reset to pending for processing")
            else:
                logger.warning(f"Failed to reset test task {self.test_task_uuid} to pending")
            
            # Step 3: Fetch task again and do normal processing
            logger.info("Test mode: Step 2 - Processing task normally...")
            task = self.api_client.fetch_next_task(reset=False)
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
                task = self.api_client.fetch_next_task(reset=effective_reset)
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
