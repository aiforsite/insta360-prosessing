"""
Video processing script for Insta360 material.
Fetches tasks from API, processes videos, and reports results.
"""

import os
import json
import time
import shutil
import logging
import requests
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv

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
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API credentials from environment
        self.api_domain = os.getenv('API_DOMAIN')
        self.api_key = os.getenv('API_KEY')
        self.media_model_dir = os.getenv('MEDIA_MODEL_DIR')
        self.stella_exec = os.getenv('STELLA_EXECUTABLE')
        self.stella_config_path = os.getenv('STELLA_CONFIG_PATH')
        self.stella_vocab_path = os.getenv('STELLA_VOCAB_PATH')
        
        if not self.api_domain or not self.api_key:
            raise ValueError("API_DOMAIN and API_KEY must be set in .env file")
        if not self.media_model_dir:
            raise ValueError("MEDIA_MODEL_DIR must be set in .env file")
        
        # Load configuration from JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.polling_interval = self.config['polling_interval']
        self.work_dir = Path(self.config['local_work_dir'])
        self.frames_per_second = self.config['frames_per_second']
        self.low_res_fps = self.config['low_res_frames_per_second']
        self.route_fps = self.config['route_calculation_fps']
        self.grace_period_days = self.config['video_deletion_grace_period_days']
        self.candidates_per_second = self.config.get('candidates_per_second', 12)  # Frames per second for selection

        self.current_task_id = None
        self.current_video_recording_id = None
        self.blur_people = False  # Can be set from task data if needed
        
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # API headers
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _api_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling."""
        url = f"{self.api_domain}{endpoint}"
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def update_status_text(self, status_text: str) -> bool:
        """Update task status_text in API."""
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}',
            json={'status_text': status_text}
        )
        return result is not None
    
    def fetch_next_task(self) -> Optional[Dict]:
        """1. Fetch next video_task from next endpoint."""
        logger.info("Fetching next video task...")
        return self._api_request('GET', '/api/v1/process-recording-task/get-next-task')
    
    def clean_local_directories(self):
        """2. Clean local directories for safety."""
        logger.info("Cleaning local work directories...")
        if self.current_task_id:
            self.update_status_text("Siivotaan paikalliset hakemistot...")
        
        if self.work_dir.exists():
            for item in self.work_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        logger.info("Local directories cleaned")
    
    def download_video(self, url: str, output_path: Path) -> bool:
        """Download video from URL."""
        try:
            logger.info(f"Downloading video from {url}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video downloaded to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return False
    
    def fetch_video_recording(self, video_recording_id: str) -> Optional[Dict]:
        """Fetch video-recording data from API."""
        logger.info(f"Fetching video-recording data for {video_recording_id}...")
        return self._api_request('GET', f'/api/v1/video-recording/{video_recording_id}/')
    
    def download_videos(self, task: Dict) -> Tuple[Optional[Path], Optional[Path]]:
        """3. Download front and back videos from video-recording data."""
        logger.info("Downloading front and back videos...")
        self.update_status_text("Ladataan front ja back videot...")
        
        # Fetch video-recording data from API
        video_recording_id = self.current_video_recording_id
        if not video_recording_id:
            logger.error("Missing video_recording_id in task")
            self.update_status_text("Virhe: video_recording_id puuttuu")
            return None, None
        
        video_recording = self.fetch_video_recording(video_recording_id)
        if not video_recording:
            logger.error("Failed to fetch video-recording data")
            self.update_status_text("Virhe: Video-recording datan haku epäonnistui")
            return None, None
        
        # Find front and back videos from videos list
        videos = video_recording.get('videos', [])
        front_video = None
        back_video = None
        
        for video in videos:
            category = video.get('category')
            if category == 'video_insta360_raw_front':
                front_video = video
            elif category == 'video_insta360_raw_back':
                back_video = video
        
        if not front_video or not back_video:
            logger.error("Missing front or back video in video-recording data")
            self.update_status_text("Virhe: Front tai back video puuttuu video-recording datasta")
            return None, None
        
        front_url = front_video.get('url')
        back_url = back_video.get('url')
        
        if not front_url or not back_url:
            logger.error("Missing video URLs in video data")
            self.update_status_text("Virhe: Videoiden URL-osoitteet puuttuvat")
            return None, None
        
        front_path = self.work_dir / "front_video.mp4"
        back_path = self.work_dir / "back_video.mp4"
        
        self.update_status_text("Ladataan front video...")
        front_ok = self.download_video(front_url, front_path)
        
        self.update_status_text("Ladataan back video...")
        back_ok = self.download_video(back_url, back_path)
        
        if front_ok and back_ok:
            self.update_status_text("Videot ladattu onnistuneesti")
            return front_path, back_path
        
        self.update_status_text("Virhe: Videoiden lataus epäonnistui")
        return None, None
    
    def stitch_videos(self, front_path: Path, back_path: Path, output_path: Path) -> bool:
        """4. Execute stitching of videos to output file using MediaSDKTest."""
        logger.info("Stitching videos...")
        self.update_status_text("Suoritetaan videoiden stitchaus...")
        try:
            import subprocess
            
            # MediaSDKTest command
            cmd = [
                '/usr/bin/MediaSDKTest',
                '-inputs', str(front_path),
                '-inputs', str(back_path),
                '-output', str(output_path),
                '-stitch_type', 'aistitch',
                '-ai_stitching_model', f'{self.media_model_dir}/ai_stitcher_v2.ins',
                '-output_size', '5760x2880',
                '-disable_cuda', 'false'
            ]
            
            logger.info(f"Running MediaSDKTest: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.info(f"MediaSDKTest output: {result.stdout}")
            if result.stderr:
                logger.warning(f"MediaSDKTest stderr: {result.stderr}")
            
            logger.info(f"Videos stitched to {output_path}")
            self.update_status_text("Stitchaus valmis")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Stitching failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            self.update_status_text(f"Virhe stitchauksessa: {e.stderr or str(e)}")
            return False
        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            self.update_status_text(f"Virhe stitchauksessa: {str(e)}")
            return False
    
    def _calculate_sharpness(self, image_path: Path) -> float:
        """Calculate Laplacian variance as sharpness metric."""
        try:
            import cv2
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image for sharpness calculation: {image_path}")
                return 0.0
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Error calculating sharpness for {image_path}: {e}")
            return 0.0
    
    def extract_frames_high_and_low(self, video_path: Path, high_dir: Path, low_dir: Path, fps: float) -> Tuple[List[Path], List[Path]]:
        """Extract high and low resolution frames simultaneously."""
        logger.info(f"Extracting {fps} FPS frames (high and low res) from {video_path}...")
        high_dir.mkdir(parents=True, exist_ok=True)
        low_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            
            # Extract high res frames (max quality)
            high_pattern = str(high_dir / "high_%06d.jpg")
            high_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps={fps}/1:round=up',
                '-q:v', '0',  # Max quality
                '-start_number', '0',
                high_pattern
            ]
            
            # Extract low res frames (3840x1920)
            low_pattern = str(low_dir / "low_%06d.jpg")
            low_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps={fps}/1:round=up',
                '-s', '3840x1920',
                '-q:v', '0',
                '-start_number', '0',
                low_pattern
            ]
            
            # Execute both commands
            result_high = subprocess.run(high_cmd, check=True, capture_output=True, text=True)
            result_low = subprocess.run(low_cmd, check=True, capture_output=True, text=True)
            
            # Collect frame paths
            high_frames = sorted(high_dir.glob("high_*.jpg"))
            low_frames = sorted(low_dir.glob("low_*.jpg"))
            
            logger.info(f"Extracted {len(high_frames)} high res and {len(low_frames)} low res frames")
            return high_frames, low_frames
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed with exit code {e.returncode}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            return [], []
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return [], []
    
    def _select_best_frames_by_sharpness(self, high_frames: List[Path], candidates_per_second: int) -> set[int]:
        """Select best frames based on sharpness from each group of candidates_per_second frames."""
        if not high_frames:
            return set()
        
        # Calculate sharpness for all high-res frames
        candidates_with_sharpness = []
        for frame_path in high_frames:
            try:
                # Extract frame number from filename (e.g., "high_000001.jpg" -> 1)
                suffix = int(frame_path.stem.split("_")[-1])
                sharpness = self._calculate_sharpness(frame_path)
                candidates_with_sharpness.append((suffix, frame_path, sharpness))
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse frame number from {frame_path}: {e}")
                continue
        
        if not candidates_with_sharpness:
            logger.warning("No high-res candidates found for selection")
            return set()
        
        # Sort by suffix
        candidates_with_sharpness.sort(key=lambda x: x[0])
        
        # Group by second and select sharpest from each group
        selected_suffixes = set()
        for i in range(0, len(candidates_with_sharpness), candidates_per_second):
            second_group = candidates_with_sharpness[i:i + candidates_per_second]
            if not second_group:
                break
            
            # Select sharpest from this group
            best_suffix, best_path, best_sharpness = max(second_group, key=lambda x: x[2])
            selected_suffixes.add(best_suffix)
            
            target_second = i // candidates_per_second
            logger.debug(f"Second {target_second}: selected frame {best_suffix} (sharpness={best_sharpness:.2f})")
        
        logger.info(f"Selected {len(selected_suffixes)} best frames from {len(candidates_with_sharpness)} candidates")
        return selected_suffixes
    
    def create_and_select_frames(self, stitched_path: Path) -> Tuple[List[Path], List[Path]]:
        """5. Extract 12fps frames (high and low res), select sharpest from each group."""
        self.update_status_text(f"Luodaan framet ({self.candidates_per_second}/s) stitchatusta tiedostosta...")
        
        high_dir = self.work_dir / "high_frames"
        low_dir = self.work_dir / "low_frames"
        
        # Extract high and low res frames
        high_frames, low_frames = self.extract_frames_high_and_low(
            stitched_path, high_dir, low_dir, self.candidates_per_second
        )
        
        if not high_frames or not low_frames:
            self.update_status_text("Virhe: Framejen luonti epäonnistui")
            return [], []
        
        self.update_status_text(f"Luotu {len(high_frames)} high res ja {len(low_frames)} low res framea")
        
        # Select best frames based on sharpness
        self.update_status_text("Valitaan terävimmät framet jokaisesta sekunnista...")
        selected_suffixes = self._select_best_frames_by_sharpness(high_frames, self.candidates_per_second)
        
        # Filter frames to only selected ones
        selected_high = [f for f in high_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        selected_low = [f for f in low_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        
        # Remove unselected frames
        for frame in high_frames:
            if frame not in selected_high:
                try:
                    frame.unlink()
                except FileNotFoundError:
                    pass
        
        for frame in low_frames:
            if frame not in selected_low:
                try:
                    frame.unlink()
                except FileNotFoundError:
                    pass
        
        # Move selected frames to final directory
        selected_dir = self.work_dir / "selected_frames"
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        final_high = []
        final_low = []
        
        for high_frame, low_frame in zip(selected_high, selected_low):
            suffix = high_frame.stem.split("_")[-1]
            new_high = selected_dir / f"high_{suffix}.jpg"
            new_low = selected_dir / f"low_{suffix}.jpg"
            
            shutil.move(str(high_frame), str(new_high))
            shutil.move(str(low_frame), str(new_low))
            
            final_high.append(new_high)
            final_low.append(new_low)
        
        self.update_status_text(f"Valittu {len(final_high)} parasta framea")
        return final_high, final_low
    
    def blur_frames_optional(self, high_frames: List[Path], low_frames: List[Path]) -> Tuple[List[Path], List[Path]]:
        """8. Optionally blur frames if blur_people is enabled."""
        if not self.blur_people:
            logger.info("Blur not enabled, skipping blur step")
            return high_frames, low_frames
        
        logger.info("Applying blur to frames...")
        self.update_status_text("Tehdään frame blurraus...")
        
        blurred_high = []
        blurred_low = []
        
        try:
            from PIL import Image, ImageFilter
            
            for frame_path in high_frames:
                img = Image.open(frame_path)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
                blurred_path = frame_path.parent / f"b_{frame_path.name}"
                blurred.save(blurred_path, quality=85)
                blurred_high.append(blurred_path)
            
            for frame_path in low_frames:
                img = Image.open(frame_path)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
                blurred_path = frame_path.parent / f"b_{frame_path.name}"
                blurred.save(blurred_path, quality=85)
                blurred_low.append(blurred_path)
            
            logger.info(f"Blurred {len(blurred_high)} high and {len(blurred_low)} low frames")
            self.update_status_text(f"Blurrattu {len(blurred_high)} framea")
            return blurred_high, blurred_low
        except Exception as e:
            logger.error(f"Frame blurring failed: {e}")
            self.update_status_text(f"Virhe frame blurrauksessa: {str(e)}")
            return high_frames, low_frames
    
    def upload_frames_to_cloud(self, frame_paths: List[Path]) -> List[Dict]:
        """9. Save frames to cloud and create frame objects."""
        logger.info("Uploading frames to cloud...")
        self.update_status_text(f"Tallennetaan {len(frame_paths)} framea pilveen ja luodaan frame objektit...")
        
        frame_objects = []
        total = len(frame_paths)
        for i, frame_path in enumerate(frame_paths):
            try:
                # Upload file
                with open(frame_path, 'rb') as f:
                    files = {'file': f}
                    data = {
                        'task_id': self.current_task_id,
                        'frame_index': i,
                        'timestamp': time.time()
                    }
                    response = requests.post(
                        f"{self.api_domain}/api/v1/process-recording-task/upload-frame",
                        headers={'Authorization': f'Token {self.api_key}'},
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    frame_obj = response.json()
                    frame_objects.append(frame_obj)
                    
                    # Update progress
                    if (i + 1) % 10 == 0 or i == total - 1:
                        self.update_status_text(f"Tallennetaan frameja pilveen: {i + 1}/{total}")
            except Exception as e:
                logger.error(f"Failed to upload frame {i}: {e}")
        
        logger.info(f"Uploaded {len(frame_objects)} frames")
        self.update_status_text(f"Tallennettu {len(frame_objects)} framea pilveen")
        return frame_objects
    
    def get_route_frames_from_low_res(self, low_frames: List[Path]) -> List[Path]:
        """10. Use 12fps low res frames for route calculation."""
        logger.info(f"Using {len(low_frames)} low res frames for route calculation...")
        self.update_status_text(f"Käytetään {len(low_frames)} low res framea reitin laskentaan...")
        return low_frames
    
    def calculate_route(self, frame_paths: List[Path]) -> Optional[Dict]:
        """11. Calculate route using selected frames (Stella VSLAM) and update raw path."""
        logger.info("Calculating route from frames using Stella VSLAM...")
        self.update_status_text(f"Lasketaan reitti valittujen frameiden avulla ({len(frame_paths)} framea)...")
        
        if not frame_paths:
            logger.warning("No frames available for route calculation")
            self.update_status_text("Virhe reitin laskennassa: frameja ei saatavilla")
            return None
        
        if not all([self.stella_exec, self.stella_config_path, self.stella_vocab_path]):
            error_msg = "Stella VSLAM environment variables (STELLA_EXECUTABLE, STELLA_CONFIG_PATH, STELLA_VOCAB_PATH) must be set"
            logger.error(error_msg)
            self.update_status_text(f"Virhe reitin laskennassa: {error_msg}")
            return None
        
        frames_dir = frame_paths[0].parent
        map_output_path = self.work_dir / "stella_map.msg"
        if map_output_path.exists():
            map_output_path.unlink()
        
        cmd = [
            self.stella_exec,
            '-c', self.stella_config_path,
            '-d', str(frames_dir),
            '--frame-skip', '1',
            '--no-sleep',
            '--auto-term',
            '--map-db-out', str(map_output_path),
            '-v', self.stella_vocab_path
        ]
        
        logger.info(f"Running Stella VSLAM command: {' '.join(cmd)}")
        try:
            import subprocess
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.info(f"Stella VSLAM output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Stella VSLAM stderr: {result.stderr}")
            
            if not map_output_path.exists():
                raise FileNotFoundError(f"Expected map output not found at {map_output_path}")
            
            route_data = {
                'map_file': str(map_output_path),
                'frame_count': len(frame_paths),
                'generated_at': datetime.now().isoformat(),
                'command': cmd
            }
            
            logger.info("Route calculated")
            self.update_status_text("Reitti laskettu, päivitetään raw path tulosten avulla...")
            return route_data
        except subprocess.CalledProcessError as e:
            logger.error(f"Stella VSLAM failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            self.update_status_text(f"Virhe reitin laskennassa: {e.stderr or str(e)}")
            return None
        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            self.update_status_text(f"Virhe reitin laskennassa: {str(e)}")
            return None

    def update_task_route(self, route_data: Dict) -> bool:
        """Update task with route data."""
        logger.info("Updating task with route data...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}/route',
            json={'raw_path': route_data}
        )
        return result is not None
    
    def mark_videos_for_deletion(self) -> bool:
        """12. Mark front and back videos for deletion after 1 month (grace period)."""
        logger.info("Marking videos for deletion...")
        self.update_status_text(f"Merkataan front ja back video tuhottavaksi {self.grace_period_days} päivän päästä (varoaika)...")
        
        deletion_date = datetime.now() + timedelta(days=self.grace_period_days)
        result = self._api_request(
            'PATCH',
            f"/api/v1/process-recording-task/{self.current_task_id}/mark-for-deletion",
            json={
                'front_video_deletion_date': deletion_date.isoformat(),
                'back_video_deletion_date': deletion_date.isoformat()
            }
        )
        if result:
            self.update_status_text("Videot merkitty poistettavaksi")
        return result is not None
    
    def report_task_completion(self, success: bool, error: Optional[str] = None):
        """Report task completion to API."""
        logger.info(f"Reporting task completion: success={success}")
        
        # Update process-recording-task status to "completed"
        task_result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}',
            json={'status': 'completed' if success else 'failed'}
        )
        
        # Update video-recording status to "ready_to_view" (if success)
        if success and self.current_video_recording_id:
            video_result = self._api_request(
                'PATCH',
                f'/api/v1/video-recording/{self.current_video_recording_id}',
                json={'status': 'ready_to_view'}
            )
            if video_result:
                logger.info("Video-recording status updated to ready_to_view")
            else:
                logger.warning("Failed to update video-recording status")
        elif not success:
            logger.info("Skipping video-recording status update due to failure")
        
        if task_result:
            logger.info(f"Process-recording-task status updated: {'completed' if success else 'failed'}")
        else:
            logger.warning("Failed to update process-recording-task status")
    
    def process_task(self, task: Dict) -> bool:
        """Process a single video task through all steps."""
        self.current_task_id = task.get('uuid')
        self.current_video_recording_id = task.get('video_recording')
        logger.info(f"Processing task {self.current_task_id}")
        self.update_status_text("Aloitetaan videokäsittelytehtävä...")
        
        try:
            # Step 2: Clean directories
            self.clean_local_directories()
            
            # Step 3: Download videos
            front_path, back_path = self.download_videos(task)
            if not front_path or not back_path:
                raise Exception("Failed to download videos")
            
            # Step 4: Stitch videos
            stitched_path = self.work_dir / "stitched_output.mp4"
            if not self.stitch_videos(front_path, back_path, stitched_path):
                raise Exception("Failed to stitch videos")
            
            # Step 5: Create and select frames (12fps high and low res, select sharpest)
            selected_high, selected_low = self.create_and_select_frames(stitched_path)
            if not selected_high or not selected_low:
                raise Exception("Failed to create and select frames")
            
            # Step 6: Optional blur
            final_high, final_low = self.blur_frames_optional(selected_high, selected_low)
            
            # Step 7: Upload frames to cloud
            frame_objects = self.upload_frames_to_cloud(final_high + final_low)
            
            # Step 8: Use 12fps low res frames for route calculation
            route_frames = self.get_route_frames_from_low_res(selected_low)
            
            # Step 9: Calculate route
            route_data = self.calculate_route(route_frames)
            if route_data:
                self.update_task_route(route_data)
            
            # Step 10: Mark videos for deletion
            self.mark_videos_for_deletion()
            
            # Step 11: Clean local directories
            self.update_status_text("Siivotaan paikalliset hakemistot...")
            self.clean_local_directories()
            
            # Report success
            self.update_status_text("Tehtävä valmis, aloitetaan uuden tehtävän pollaus...")
            self.report_task_completion(success=True)
            logger.info(f"Task {self.current_task_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.update_status_text(f"Virhe: {str(e)}")
            self.report_task_completion(success=False, error=str(e))
            self.clean_local_directories()
            return False
    
    def run(self):
        """Main loop: poll for tasks and process them."""
        logger.info("Starting video processor...")
        logger.info(f"Polling interval: {self.polling_interval} seconds")
        
        while True:
            try:
                # Step 1: Fetch next task
                task = self.fetch_next_task()
                task_id = task.get('uuid')
                if task and (task_id):
                    logger.info(f"Found task: {task_id}")
                    self.process_task(task)
                else:
                    logger.debug("No tasks available, waiting...")
                
                # Wait before next poll
                time.sleep(self.polling_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.polling_interval)


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()

