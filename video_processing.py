"""
Video processing module for downloading and stitching videos.
"""

import logging
import os
import re
import subprocess
import requests
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class VideoProcessing:
    """Handles video download and stitching operations."""
    
    def __init__(self, work_dir: Path, mediasdk_executable: str):
        """Initialize video processing."""
        self.work_dir = work_dir
        self.mediasdk_executable = mediasdk_executable
    
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
    
    def _infer_file_extension(self, url: str, default_extension: str = ".mp4") -> str:
        """Infer file extension from URL query/path (handles S3 response-content-disposition)."""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            # Prefer filename from response-content-disposition parameter because S3 paths are hash-like
            rcd_values = query_params.get('response-content-disposition') or []
            for value in rcd_values:
                decoded = unquote(value)
                match = re.search(r'filename\s*=\s*"?([^";]+)"?', decoded, re.IGNORECASE)
                if match:
                    suffix = Path(match.group(1)).suffix
                    if suffix:
                        return suffix
            
            # Fallback to direct filename parameter if present
            filename_values = query_params.get('filename') or []
            for value in filename_values:
                suffix = Path(unquote(value)).suffix
                if suffix:
                    return suffix
            
            # Finally, try to use suffix from actual URL path
            path_suffix = Path(unquote(parsed.path)).suffix
            if path_suffix:
                return path_suffix
        except Exception as exc:
            logger.debug(f"Could not infer extension from URL {url}: {exc}")
        
        return default_extension
    
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
        
        front_ext = self._infer_file_extension(front_url, ".insv")
        back_ext = self._infer_file_extension(back_url, ".insv")
        
        front_path = self.work_dir / f"front_video{front_ext}"
        back_path = self.work_dir / f"back_video{back_ext}"
        
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
        
        # Verify input files exist and have content
        if not front_path.exists():
            logger.error(f"Front video not found: {front_path}")
            update_status_callback(f"Virhe: Front video ei löydy: {front_path}")
            return False
        
        if not back_path.exists():
            logger.error(f"Back video not found: {back_path}")
            update_status_callback(f"Virhe: Back video ei löydy: {back_path}")
            return False
        
        front_size = front_path.stat().st_size
        back_size = back_path.stat().st_size
        logger.info(f"Input files - Front: {front_path.name} ({front_size:,} bytes), Back: {back_path.name} ({back_size:,} bytes)")
        
        try:
            # MediaSDKTest command for Windows
            cmd = [
                self.mediasdk_executable,
                '-inputs', str(front_path),
                '-inputs', str(back_path),
                '-output_size', '5760x2880',
                '-enable_flowstate', 'ON',
                '-enable_directionlock', 'ON',
                '-output', str(output_path)
            ]
            # Use environment variables as-is (Windows doesn't need LD_LIBRARY_PATH)
            env = os.environ.copy()
            
            logger.info(f"Running MediaSDKTest: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.stdout:
                logger.info(f"MediaSDKTest output: {result.stdout}")
            if result.stderr:
                logger.warning(f"MediaSDKTest stderr: {result.stderr}")
            
            # Verify output file was created and has content
            if not output_path.exists():
                logger.error(f"Stitched output file not created: {output_path}")
                update_status_callback("Virhe: Stitchattu video ei luotu")
                return False
            
            output_size = output_path.stat().st_size
            logger.info(f"Stitched output: {output_path.name} ({output_size:,} bytes)")
            
            if output_size == 0:
                logger.error(f"Stitched output file is empty: {output_path}")
                update_status_callback("Virhe: Stitchattu video on tyhjä")
                return False
            
            # Try to get video properties using ffprobe if available
            try:
                ffprobe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,pix_fmt,codec_name',
                    '-of', 'json',
                    str(output_path)
                ]
                ffprobe_result = subprocess.run(
                    ffprobe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if ffprobe_result.returncode == 0:
                    import json
                    probe_data = json.loads(ffprobe_result.stdout)
                    streams = probe_data.get('streams', [])
                    if streams:
                        stream = streams[0]
                        width = stream.get('width')
                        height = stream.get('height')
                        codec = stream.get('codec_name')
                        pix_fmt = stream.get('pix_fmt')
                        logger.info(f"Stitched video properties: {width}x{height}, codec={codec}, pix_fmt={pix_fmt}")
                        if width and height:
                            aspect_ratio = width / height
                            logger.info(f"Aspect ratio: {aspect_ratio:.2f} (expected 2.0 for equirectangular 360)")
                            if abs(aspect_ratio - 2.0) > 0.1:
                                logger.warning(f"Unexpected aspect ratio: {aspect_ratio:.2f}, expected 2.0 for equirectangular 360")
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
                logger.debug(f"Could not probe video with ffprobe: {e}")
            
            logger.info(f"Videos stitched successfully to {output_path}")
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

