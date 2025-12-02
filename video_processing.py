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
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


class VideoProcessing:
    """Handles video download and stitching operations."""
    
    def __init__(self, work_dir: Path, mediasdk_executable: str, stitch_config: Optional[Dict] = None):
        """Initialize video processing."""
        self.work_dir = work_dir
        self.mediasdk_executable = mediasdk_executable
        self.stitch_config = stitch_config or {}
        # Default stitch parameters
        self.output_size = self.stitch_config.get('output_size', '5760x2880')
        self.enable_flowstate = self.stitch_config.get('enable_flowstate', 'ON')
        self.enable_directionlock = self.stitch_config.get('enable_directionlock', 'ON')
        self.disable_cuda = self.stitch_config.get('disable_cuda', False)  # false = enable CUDA
        self.enable_debug_info = self.stitch_config.get('enable_debug_info', 'ON')
        # Optional advanced parameters
        self.stitch_type = self.stitch_config.get('stitch_type', None)  # template, optflow, dynamicstitch, aistitch
        self.ai_stitching_model = self.stitch_config.get('ai_stitching_model', '')
        self.enable_stitchfusion = self.stitch_config.get('enable_stitchfusion', 'OFF')
        self.enable_denoise = self.stitch_config.get('enable_denoise', 'OFF')
        self.enable_colorplus = self.stitch_config.get('enable_colorplus', 'OFF')
        self.enable_deflicker = self.stitch_config.get('enable_deflicker', 'OFF')
        self.image_processing_accel = self.stitch_config.get('image_processing_accel', None)  # auto, cpu
        self.bitrate = self.stitch_config.get('bitrate', None)  # None = same as input
        self.enable_h265_encoder = self.stitch_config.get('enable_h265_encoder', False)
    
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
        update_status_callback("Downloading front and back videos...")
        
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
            update_status_callback("Error: Front or back video missing from video-recording data")
            return None, None
        
        front_url = front_video.get('url')
        back_url = back_video.get('url')
        
        if not front_url or not back_url:
            logger.error("Missing video URLs in video data")
            update_status_callback("Error: Video URL addresses missing")
            return None, None
        
        front_ext = self._infer_file_extension(front_url, ".insv")
        back_ext = self._infer_file_extension(back_url, ".insv")
        
        front_path = self.work_dir / f"front_video{front_ext}"
        back_path = self.work_dir / f"back_video{back_ext}"
        
        update_status_callback("Downloading front video...")
        front_ok = self.download_video(front_url, front_path)
        
        update_status_callback("Downloading back video...")
        back_ok = self.download_video(back_url, back_path)
        
        if front_ok and back_ok:
            update_status_callback("Videos downloaded successfully")
            return front_path, back_path
        
        update_status_callback("Error: Video download failed")
        return None, None
    
    def stitch_videos(self, front_path: Path, back_path: Path, output_path: Path, update_status_callback) -> bool:
        """Execute stitching of videos to output file using MediaSDKTest."""
        logger.info("Stitching videos...")
        update_status_callback("Stitching videos...")
        
        # Verify input files exist and have content
        if not front_path.exists():
            logger.error(f"Front video not found: {front_path}")
            update_status_callback(f"Error: Front video not found: {front_path}")
            return False
        
        if not back_path.exists():
            logger.error(f"Back video not found: {back_path}")
            update_status_callback(f"Error: Back video not found: {back_path}")
            return False
        
        front_size = front_path.stat().st_size
        back_size = back_path.stat().st_size
        logger.info(f"Input files - Front: {front_path.name} ({front_size:,} bytes), Back: {back_path.name} ({back_size:,} bytes)")
        
        try:
            # MediaSDKTest command for Windows
            # Based on correct command format:
            # MediaSDKTest.exe -inputs front.insv back.insv -disable_cuda false -enable_debug_info ON -output_size 5760x2880 -enable_directionlock ON -enable_flowstate ON -output output.mp4
            cmd = [
                self.mediasdk_executable,
                '-inputs', str(front_path), str(back_path),  # Both inputs after single -inputs parameter
                '-disable_cuda', 'false' if not self.disable_cuda else 'true',
                '-enable_debug_info', self.enable_debug_info,
                '-output_size', self.output_size,
                '-enable_directionlock', self.enable_directionlock,
                '-enable_flowstate', self.enable_flowstate,
                '-output', str(output_path)
            ]
            
            # Add optional parameters if configured
            if self.stitch_type:
                cmd.extend(['-stitch_type', self.stitch_type])
            
            if self.ai_stitching_model and self.stitch_type == 'aistitch':
                cmd.extend(['-ai_stitching_model', self.ai_stitching_model])
            
            if self.image_processing_accel:
                cmd.extend(['-image_processing_accel', self.image_processing_accel])
            
            if self.enable_stitchfusion != 'OFF':
                cmd.extend(['-enable_stitchfusion', self.enable_stitchfusion])
            
            if self.enable_denoise != 'OFF':
                cmd.extend(['-enable_denoise', self.enable_denoise])
            
            if self.enable_colorplus != 'OFF':
                cmd.extend(['-enable_colorplus', self.enable_colorplus])
            
            if self.enable_deflicker != 'OFF':
                cmd.extend(['-enable_deflicker', self.enable_deflicker])
            
            if self.bitrate:
                cmd.extend(['-bitrate', str(self.bitrate)])
            
            if self.enable_h265_encoder:
                cmd.extend(['-enable_h265_encoder'])
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
                update_status_callback("Error: Stitched video was not created")
                return False
            
            output_size = output_path.stat().st_size
            logger.info(f"Stitched output: {output_path.name} ({output_size:,} bytes)")
            
            if output_size == 0:
                logger.error(f"Stitched output file is empty: {output_path}")
                update_status_callback("Error: Stitched video is empty")
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
            
            # Add 360 metadata to the stitched video using ffmpeg
            # MediaSDKTest.exe may not add proper 360 metadata, so we add it here
            # This ensures the video is recognized as 360-degree equirectangular by players
            logger.info("Adding 360 metadata to stitched video...")
            update_status_callback("Lisätään 360-metadataa videoon...")
            
            temp_output = output_path.with_suffix('.temp' + output_path.suffix)
            try:
                # Use ffmpeg to inject proper 360 metadata
                # First try with copy mode (fast, but may not work for all containers)
                ffmpeg_cmd_copy = [
                    'ffmpeg',
                    '-i', str(output_path),
                    '-c', 'copy',  # Copy streams without re-encoding
                    '-metadata', 'spherical-video=1',
                    '-metadata', 'stereo-mode=mono',
                    '-y',  # Overwrite output file
                    str(temp_output)
                ]
                
                ffmpeg_result = subprocess.run(
                    ffmpeg_cmd_copy,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if ffmpeg_result.returncode == 0:
                    # Verify the temp file was created
                    if temp_output.exists() and temp_output.stat().st_size > 0:
                        output_path.unlink()
                        temp_output.rename(output_path)
                        logger.info("360 metadata added successfully (copy mode)")
                        update_status_callback("360-metadata lisätty")
                    else:
                        logger.warning("Copy mode produced empty file, trying re-encode")
                        if temp_output.exists():
                            temp_output.unlink()
                        raise Exception("Copy mode failed")
                else:
                    logger.warning(f"Copy mode failed: {ffmpeg_result.stderr}")
                    if temp_output.exists():
                        temp_output.unlink()
                    raise Exception("Copy mode failed")
                    
            except Exception as e:
                logger.info(f"Trying re-encode mode to add 360 metadata: {e}")
                try:
                    # Re-encode with proper 360 metadata injection
                    # This ensures metadata is properly embedded
                    ffmpeg_cmd_reencode = [
                        'ffmpeg',
                        '-i', str(output_path),
                        '-vf', 'scale=5760:2880:flags=lanczos',  # Ensure correct resolution
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '18',  # High quality
                        '-pix_fmt', 'yuv420p',
                        '-metadata', 'spherical-video=1',
                        '-metadata', 'stereo-mode=mono',
                        '-movflags', '+faststart',  # Web optimization
                        '-y',
                        str(temp_output)
                    ]
                    
                    ffmpeg_result = subprocess.run(
                        ffmpeg_cmd_reencode,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if ffmpeg_result.returncode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
                        output_path.unlink()
                        temp_output.rename(output_path)
                        logger.info("360 metadata added successfully (re-encode mode)")
                        update_status_callback("360-metadata lisätty")
                    else:
                        logger.warning(f"Re-encode mode failed: {ffmpeg_result.stderr if ffmpeg_result else 'No output'}")
                        if temp_output.exists():
                            temp_output.unlink()
                        update_status_callback("Varoitus: 360-metadataa ei voitu lisätä, käytetään alkuperäistä videota")
                except (subprocess.TimeoutExpired, FileNotFoundError) as e2:
                    logger.warning(f"Could not add 360 metadata (ffmpeg not available or timeout): {e2}")
                    if temp_output.exists():
                        temp_output.unlink()
                    update_status_callback("Varoitus: 360-metadataa ei voitu lisätä")
            
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
            update_status_callback(f"Error in stitching: {str(e)}")
            return False
    
    def store_processed_video(
        self,
        stitched_path: Path,
        project_id: str,
        video_recording: Optional[Dict],
        api_client,
        video_type: str = 'video_insta360_processed_stitched'
    ) -> Optional[str]:
        """Upload stitched video binary to API and return video UUID."""
        if not stitched_path.exists():
            logger.warning(f"Processed video not found at {stitched_path}")
            return None
        try:
            with open(stitched_path, 'rb') as f:
                video_binary = f.read()
            video_name = stitched_path.name
            video_recording_id = video_recording.get('uuid') if video_recording else api_client.current_video_recording_id
            video_id = api_client.store_video(
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
    
    def select_fallback_video_id(
        self,
        video_recording: Optional[Dict],
        fallback_categories: Optional[List[str]] = None
    ) -> Optional[str]:
        """Choose an existing video UUID to associate frames if processed video missing."""
        if not video_recording:
            return None
        videos = video_recording.get('videos', []) or []
        if fallback_categories is None:
            fallback_categories = [
                'video_insta360_raw_front',
                'video_insta360_raw_back'
            ]
        for category in fallback_categories:
            for video in videos:
                if video.get('category') == category and video.get('uuid'):
                    return video.get('uuid')
        if videos:
            return videos[0].get('uuid')
        return None

