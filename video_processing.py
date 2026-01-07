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
        self.stitch_type = self.stitch_config.get('stitch_type', 'dynamicstitch')  # template, optflow, dynamicstitch, aistitch
        self.ai_stitching_model = self.stitch_config.get('ai_stitching_model', '')
        self.enable_stitchfusion = self.stitch_config.get('enable_stitchfusion', 'OFF')
        self.enable_denoise = self.stitch_config.get('enable_denoise', 'OFF')
        self.enable_colorplus = self.stitch_config.get('enable_colorplus', 'OFF')
        self.enable_deflicker = self.stitch_config.get('enable_deflicker', 'OFF')
        self.image_processing_accel = self.stitch_config.get('image_processing_accel', None)  # auto, cpu
        self.bitrate = self.stitch_config.get('bitrate', None)  # None = same as input
        self.enable_h265_encoder = self.stitch_config.get('enable_h265_encoder', False)
    
    def download_video(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """Download video from URL with retry logic, resume support, and improved error handling.
        
        Args:
            url: Video URL to download
            output_path: Path where video will be saved
            max_retries: Maximum number of retry attempts (default: 3)
        
        Returns:
            True if download successful, False otherwise
        """
        import time
        from requests.exceptions import RequestException, Timeout, ConnectionError, ChunkedEncodingError
        from requests.adapters import HTTPAdapter
        try:
            from urllib3.util.retry import Retry
        except ImportError:
            # Fallback if urllib3 version doesn't have Retry
            Retry = None
        
        # Create session with retry strategy for connection pooling
        session = requests.Session()
        
        # Configure retry strategy for connection issues if available
        if Retry is not None:
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        else:
            # Fallback: use basic HTTPAdapter without retry strategy
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        
        try:
            for attempt in range(max_retries):
                try:
                    # Check if partial file exists for resume
                    resume_pos = 0
                    if output_path.exists() and attempt > 0:
                        resume_pos = output_path.stat().st_size
                        logger.info(f"Resuming download from byte {resume_pos:,}")
                    elif attempt > 0 and output_path.exists():
                        # Remove partial file only if it's too small (likely corrupted)
                        partial_size = output_path.stat().st_size
                        if partial_size < 1024 * 1024:  # Less than 1MB, likely corrupted
                            logger.info(f"Retry {attempt + 1}/{max_retries}: Removing small partial file {output_path}")
                            output_path.unlink()
                        else:
                            resume_pos = partial_size
                            logger.info(f"Resuming download from byte {resume_pos:,}")
                    
                    logger.info(f"Downloading video from {url} (attempt {attempt + 1}/{max_retries})...")
                    
                    # Prepare headers for resume support
                    headers = {
                        'Accept-Encoding': 'identity',  # Disable compression for binary data
                        'Connection': 'keep-alive',  # Keep connection alive
                    }
                    if resume_pos > 0:
                        headers['Range'] = f'bytes={resume_pos}-'
                    
                    # Use longer timeouts for large files
                    # connect timeout: 60s, read timeout: 1800s (30 minutes total)
                    response = session.get(
                        url,
                        stream=True,
                        timeout=(60, 1800),  # Increased timeouts
                        headers=headers,
                        allow_redirects=True
                    )
                    
                    # Handle partial content (206) for resume
                    if response.status_code == 206:
                        logger.info(f"Server supports resume: continuing from byte {resume_pos:,}")
                    elif response.status_code == 200 and resume_pos > 0:
                        # Server doesn't support resume, need to restart
                        logger.warning("Server doesn't support resume, restarting download")
                        if output_path.exists():
                            output_path.unlink()
                        resume_pos = 0
                        # Retry without Range header
                        response = session.get(
                            url,
                            stream=True,
                            timeout=(60, 1800),
                            headers={'Accept-Encoding': 'identity', 'Connection': 'keep-alive'},
                            allow_redirects=True
                        )
                    
                    response.raise_for_status()
                    
                    # Get expected file size
                    total_size = None
                    if 'content-range' in response.headers:
                        # Parse Content-Range: bytes 869253120-1090519039/1090519040
                        content_range = response.headers['content-range']
                        total_size = int(content_range.split('/')[-1])
                        logger.info(f"Resuming: {resume_pos:,} - {total_size:,} bytes (total: {total_size:,} bytes)")
                    elif 'content-length' in response.headers:
                        content_length = int(response.headers['content-length'])
                        if resume_pos > 0:
                            total_size = resume_pos + content_length  # Adjust for resume
                        else:
                            total_size = content_length
                        logger.info(f"Expected file size: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
                    
                    # Use smaller chunk size for more reliable downloads (512KB chunks)
                    # Smaller chunks = less data lost if connection breaks
                    chunk_size = 512 * 1024  # 512 KB
                    downloaded = resume_pos
                    
                    # Open file in append mode if resuming, otherwise write mode
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    with open(output_path, mode) as f:
                        try:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:  # Filter out keep-alive chunks
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    
                                    # Log progress for large files
                                    if total_size and downloaded % (50 * 1024 * 1024) == 0:  # Every 50 MB
                                        progress = (downloaded / total_size) * 100
                                        logger.info(f"Download progress: {downloaded:,} / {total_size:,} bytes ({progress:.1f}%)")
                            
                            # Flush to disk
                            f.flush()
                            os.fsync(f.fileno())
                            
                        except (ChunkedEncodingError, ConnectionError) as chunk_error:
                            # If chunked encoding fails, we have partial data
                            logger.warning(f"Chunked encoding error during download: {chunk_error}")
                            # File is already partially written, will resume on next attempt
                            raise
                    
                    # Verify file was downloaded completely
                    if total_size and output_path.stat().st_size != total_size:
                        actual_size = output_path.stat().st_size
                        if actual_size < total_size:
                            raise Exception(f"File incomplete: expected {total_size:,} bytes, got {actual_size:,} bytes ({total_size - actual_size:,} bytes missing)")
                        else:
                            logger.warning(f"File size larger than expected: {total_size:,} vs {actual_size:,} bytes")
                    
                    file_size = output_path.stat().st_size
                    logger.info(f"Video downloaded successfully to {output_path} ({file_size:,} bytes, {file_size / (1024*1024):.2f} MB)")
                    return True
                    
                except (ChunkedEncodingError, ConnectionError, Timeout) as e:
                    is_last_attempt = (attempt == max_retries - 1)
                    error_msg = f"Download failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    
                    if is_last_attempt:
                        logger.error(error_msg)
                        if output_path.exists():
                            partial_size = output_path.stat().st_size
                            logger.warning(f"Partial file exists: {output_path} ({partial_size:,} bytes, {partial_size / (1024*1024):.2f} MB)")
                        return False
                    else:
                        logger.warning(error_msg)
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + (attempt * 2)  # 2s, 4s, 8s
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        
                except RequestException as e:
                    # For other request errors, don't retry (e.g., 404, 403)
                    logger.error(f"Download failed: {type(e).__name__}: {e}")
                    if output_path.exists() and attempt == 0:  # Only delete on first attempt for HTTP errors
                        output_path.unlink()
                    return False
                    
                except Exception as e:
                    logger.error(f"Unexpected error during download: {type(e).__name__}: {e}")
                    if output_path.exists() and attempt == max_retries - 1:
                        # Keep partial file on last attempt for debugging
                        pass
                    elif output_path.exists() and attempt > 0:
                        # Keep partial file for resume on retry
                        pass
                    if attempt == max_retries - 1:
                        return False
                    wait_time = (2 ** attempt) + (attempt * 2)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        finally:
            # Close session
            session.close()
        
        return False
    
    def get_video_duration(self, video_path: Path) -> Optional[float]:
        """
        Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Duration in seconds, or None if unable to determine
        """
        try:
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                duration = probe_data.get('format', {}).get('duration')
                if duration:
                    duration_float = float(duration)
                    logger.info(f"Video duration: {duration_float:.2f} seconds ({duration_float / 60:.2f} minutes)")
                    return duration_float
            logger.warning(f"Could not extract duration from ffprobe output")
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffprobe not available or timeout: {e}")
            return None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse ffprobe output: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting video duration: {e}")
            return None
    
    def get_video_creation_time(self, video_path: Path) -> Optional[str]:
        """
        Get video creation time from metadata using ffprobe.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Creation time in ISO format (YYYY-MM-DDTHH:MM:SS), or None if unable to determine
        """
        try:
            # Try to get creation_time from format tags
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format_tags=creation_time',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                # Try to get creation_time from format tags
                # ffprobe returns: {"format": {"tags": {"creation_time": "2025-08-19T09:18:07.000000Z"}}}
                creation_time = probe_data.get('format', {}).get('tags', {}).get('creation_time')
                
                if not creation_time:
                    # Fallback: try to get from format level (some videos might have it there)
                    creation_time = probe_data.get('format', {}).get('creation_time')
                
                if creation_time:
                    # ffprobe returns creation_time in format: 2024-01-15T10:30:45.000000Z
                    # Convert to ISO format without microseconds and timezone
                    # Parse and reformat to ensure proper ISO format
                    from datetime import datetime
                    try:
                        # Try parsing with microseconds and timezone
                        dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                        # Format to ISO without microseconds: YYYY-MM-DDTHH:MM:SS
                        iso_time = dt.strftime('%Y-%m-%dT%H:%M:%S')
                        logger.info(f"Video creation time: {iso_time}")
                        return iso_time
                    except ValueError:
                        # Fallback: try parsing without timezone
                        try:
                            dt = datetime.fromisoformat(creation_time.split('.')[0])
                            iso_time = dt.strftime('%Y-%m-%dT%H:%M:%S')
                            logger.info(f"Video creation time: {iso_time}")
                            return iso_time
                        except ValueError:
                            logger.warning(f"Could not parse creation_time format: {creation_time}")
                            return None
            
            # Fallback: try to get from file modification time if metadata not available
            logger.warning(f"Could not extract creation_time from video metadata, trying file modification time")
            try:
                from datetime import datetime
                import os
                mtime = os.path.getmtime(video_path)
                dt = datetime.fromtimestamp(mtime)
                iso_time = dt.strftime('%Y-%m-%dT%H:%M:%S')
                logger.info(f"Using file modification time as creation time: {iso_time}")
                return iso_time
            except Exception as e:
                logger.warning(f"Could not get file modification time: {e}")
                return None
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffprobe not available or timeout: {e}")
            return None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse ffprobe output: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error getting video creation time: {e}")
            return None
    
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
    
    def _generate_frame_indices(self, video_duration_seconds: float, source_fps: float = 29.95, target_fps: float = 10.0) -> List[int]:
        """
        Generate frame indices for frame extraction.
        
        Args:
            video_duration_seconds: Video duration in seconds
            source_fps: Source video FPS (default: 29.95)
            target_fps: Target FPS for frame extraction (default: 10.0)
        
        Returns:
            List of frame indices (e.g., [0, 3, 6, 9, ...] for every 3rd frame)
        """
        # Calculate frame interval (every Nth frame)
        frame_interval = int(round(source_fps / target_fps))
        if frame_interval < 1:
            frame_interval = 1
        
        # Calculate total number of frames in source video
        total_frames = int(video_duration_seconds * source_fps)
        
        # Generate indices: every Nth frame
        indices = []
        current_index = 0
        while current_index < total_frames:
            indices.append(current_index)
            current_index += frame_interval
        
        logger.info(f"Generated {len(indices)} frame indices for {video_duration_seconds:.2f}s video "
                   f"({source_fps} fps -> {target_fps} fps, interval={frame_interval})")
        return indices
    
    def extract_frames_direct(self, front_path: Path, back_path: Path, frames_dir: Path, frame_indices: List[int], update_status_callback) -> bool:
        """
        Extract frames directly from videos using MediaSDKTest without creating intermediate video.
        
        Args:
            front_path: Path to front video file
            back_path: Path to back video file
            frames_dir: Directory where frames will be saved
            frame_indices: List of frame indices to extract (e.g., [0, 3, 6, 9, ...])
            update_status_callback: Status update callback
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Extracting frames directly from videos using MediaSDKTest...")
        update_status_callback("Extracting frames directly from videos...")
        
        # Verify input files exist and have content
        if not front_path.exists():
            logger.error(f"Front video not found: {front_path}")
            update_status_callback(f"Error: Front video not found: {front_path}")
            return False
        
        if not back_path.exists():
            logger.error(f"Back video not found: {back_path}")
            update_status_callback(f"Error: Back video not found: {back_path}")
            return False
        
        # Create frames directory
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Format frame indices as string: "0-3-6-9-12-..."
        frame_indices_str = '-'.join(str(idx) for idx in frame_indices)
        
        front_size = front_path.stat().st_size
        back_size = back_path.stat().st_size
        logger.info(f"Input files - Front: {front_path.name} ({front_size:,} bytes), Back: {back_path.name} ({back_size:,} bytes)")
        logger.info(f"Extracting {len(frame_indices)} frames to {frames_dir}")
        
        try:
            # MediaSDKTest command to extract frames directly
            # Based on: MediaSDKTest.exe -inputs front.insv back.insv ... -image_sequence_dir dir -image_type jpg -export_frame_index 0-3-6-9-...
            cmd = [
                self.mediasdk_executable,
                '-inputs', str(front_path), str(back_path),
                '-disable_cuda', 'false' if not self.disable_cuda else 'true',
                '-enable_h265_encoder', 'h265',
                '-output_size', self.output_size,
                '-enable_directionlock', self.enable_directionlock,
                '-enable_flowstate', self.enable_flowstate,
                '-stitch_type', self.stitch_type,
                '-image_sequence_dir', str(frames_dir),
                '-image_type', 'jpg',
                '-export_frame_index', frame_indices_str
            ]
            
            # Add optional parameters if configured
            if self.enable_debug_info == "ON":
                cmd.extend(['-enable_debug_info', 'ON'])
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
            
            # Use environment variables as-is (Windows doesn't need LD_LIBRARY_PATH)
            env = os.environ.copy()
            
            logger.info(f"Running MediaSDKTest for frame extraction: {' '.join(cmd)}")
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
            
            # Verify frames were created
            frame_files = list(frames_dir.glob("*.jpg"))
            if not frame_files:
                logger.error(f"No frames were created in {frames_dir}")
                update_status_callback("Error: No frames were extracted")
                return False
            
            logger.info(f"Extracted {len(frame_files)} frames successfully")
            update_status_callback(f"Extracted {len(frame_files)} frames")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            
            update_status_callback(f"Error in frame extraction: {e.stderr or str(e)}")
            return False
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            update_status_callback(f"Error in frame extraction: {str(e)}")
            return False
    
    def stitch_videos(self, front_path: Path, back_path: Path, output_path: Path, update_status_callback) -> bool:
        """Execute stitching of videos to H265 MP4 output file using MediaSDKTest.
        
        Creates an intermediate H265-encoded MP4 video that can be used for frame extraction with FFmpeg.
        """
        # Keep old implementation for backward compatibility if needed
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
                '-output_size', self.output_size,
                '-enable_directionlock', self.enable_directionlock,
                '-enable_flowstate', self.enable_flowstate,
                '-output', str(output_path)
            ]
            
            # Add optional parameters if configured
            if self.enable_debug_info == "ON":
                cmd.extend(['-enable_debug_info', 'ON'])

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
            
            # Always use H265 encoder for intermediate video
            if self.enable_h265_encoder:
                cmd.extend(['-enable_h265_encoder', 'h265'])
            else:
                # Force H265 even if not explicitly enabled in config
                cmd.extend(['-enable_h265_encoder', 'h265'])
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
            
            logger.info(f"Videos stitched successfully to {output_path}")
            update_status_callback("Stitching complete")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Stitching failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            
            update_status_callback(f"Error in stitching: {e.stderr or str(e)}")
            return False
        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            update_status_callback(f"Error in stitching: {str(e)}")
            return False
    
    def store_processed_video(
        self,
        stitched_path: Optional[Path],
        project_id: str,
        video_recording: Optional[Dict],
        api_client,
        video_type: str = 'video_insta360_processed_stitched',
        video_duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Create video object in API without uploading binary.
        
        Args:
            stitched_path: Optional path to stitched video (not uploaded, kept for reference)
            project_id: Project UUID
            video_recording: Video recording dict
            api_client: API client instance
            video_type: Video type/category
            video_duration: Optional video duration in seconds
        
        Returns:
            Video UUID if successful, None otherwise
        """
        try:
            video_name = stitched_path.name if stitched_path and stitched_path.exists() else 'stitched_video.mp4'
            video_recording_id = video_recording.get('uuid') if video_recording else api_client.current_video_recording_id
            
            # Create video object without uploading binary
            # We'll use a minimal size (0) since we're not uploading the binary
            video_id = api_client.store_video(
                project_id=project_id,
                video_recording_id=video_recording_id,
                video_type=video_type,
                video_size=0,  # No binary to upload
                video_binary=b'',  # Empty binary
                name=video_name
            )
            if video_id:
                logger.info(f"Created processed video object with ID {video_id} (no binary uploaded)")
            return video_id
        except Exception as exc:
            logger.error(f"Failed to create processed video object: {exc}")
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

