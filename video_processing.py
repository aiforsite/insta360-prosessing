"""
Video processing module for downloading and stitching videos.
"""

import logging
import subprocess
import requests
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class VideoProcessing:
    """Handles video download and stitching operations."""
    
    def __init__(self, work_dir: Path, media_model_dir: str):
        """Initialize video processing."""
        self.work_dir = work_dir
        self.media_model_dir = media_model_dir
    
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
    
    def download_videos(self, video_recording: Dict, api_client, update_status_callback) -> Tuple[Optional[Path], Optional[Path]]:
        """Download front and back videos from video-recording data."""
        logger.info("Downloading front and back videos...")
        update_status_callback("Ladataan front ja back videot...")
        
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
            update_status_callback("Virhe: Front tai back video puuttuu video-recording datasta")
            return None, None
        
        front_url = front_video.get('url')
        back_url = back_video.get('url')
        
        if not front_url or not back_url:
            logger.error("Missing video URLs in video data")
            update_status_callback("Virhe: Videoiden URL-osoitteet puuttuvat")
            return None, None
        
        front_path = self.work_dir / "front_video.mp4"
        back_path = self.work_dir / "back_video.mp4"
        
        update_status_callback("Ladataan front video...")
        front_ok = self.download_video(front_url, front_path)
        
        update_status_callback("Ladataan back video...")
        back_ok = self.download_video(back_url, back_path)
        
        if front_ok and back_ok:
            update_status_callback("Videot ladattu onnistuneesti")
            return front_path, back_path
        
        update_status_callback("Virhe: Videoiden lataus epäonnistui")
        return None, None
    
    def stitch_videos(self, front_path: Path, back_path: Path, output_path: Path, update_status_callback) -> bool:
        """Execute stitching of videos to output file using MediaSDKTest."""
        logger.info("Stitching videos...")
        update_status_callback("Suoritetaan videoiden stitchaus...")
        try:
            # MediaSDKTest command
            cmd = [
                '/usr/bin/MediaSDKTest',
                '-inputs', str(front_path),
                '-inputs', str(back_path),
                '-output', str(output_path),
                #'-stitch_type', 'aistitch',
                # '-ai_stitching_model', f'{self.media_model_dir}/ai_stitcher_v2.ins',
                "-enable_flowstate",
                "-enable_directionlock",
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
            update_status_callback("Stitchaus valmis")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Stitching failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            update_status_callback(f"Virhe stitchauksessa: {e.stderr or str(e)}")
            return False
        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            update_status_callback(f"Virhe stitchauksessa: {str(e)}")
            return False

