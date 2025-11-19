"""
Video processing module for downloading and stitching videos.
"""

import logging
import os
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
                '-stitch_type', 'aistitch',
                 '-ai_stitching_model', f'{self.media_model_dir}/ai_stitcher_v2.ins',
                "-enable_flowstate",
                "-enable_directionlock",
                '-output_size', '5760x2880',
                '-disable_cuda', 'false'
            ]
            
            # Aseta LD_LIBRARY_PATH varmistaaksesi, että CUDA 11.x kirjastot löytyvät
            cuda11_lib_path = '/usr/local/cuda-11/targets/x86_64-linux/lib'
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if cuda11_lib_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda11_lib_path}:{current_ld_path}" if current_ld_path else cuda11_lib_path
                logger.info(f"Set LD_LIBRARY_PATH to include CUDA 11.x libraries: {cuda11_lib_path}")
            
            # Varmista, että subprocess käyttää päivitettyä ympäristöä
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
                
                stderr_lower = result.stderr.lower()
                
                # Check if CUDA/GPU is not being used
                cuda_warnings = [
                    'cudacontext null',
                    'cuda context null',
                    'cuda not available',
                    'gpu not available',
                    'no cuda device',
                    'cuda initialization failed'
                ]
                
                cuda_not_used = any(warning in stderr_lower for warning in cuda_warnings)
                if cuda_not_used:
                    logger.warning("⚠️  CUDA/GPU is not being used for stitching! Stitching may be slower or incorrect.")
                    logger.warning("   Check GPU drivers, CUDA installation, and MediaSDKTest configuration.")
                    update_status_callback("Varoitus: GPU ei ole käytössä stitchauksessa")
                
                # Check for mask file error
                if 'can\'t find input: mask' in stderr_lower or 'can\'t find input mask' in stderr_lower:
                    logger.warning("⚠️  Mask file is missing! MediaSDKTest expects a mask file for AI stitching.")
                    logger.warning("   This may cause flow estimator to fail and result in incorrect stitching.")
                    update_status_callback("Varoitus: Mask-tiedosto puuttuu")
                
                # Check for flow estimator errors
                if 'failed to create flow estimator' in stderr_lower or 'flow estimator' in stderr_lower:
                    logger.error("❌ Flow estimator creation failed! AI stitching may not work correctly.")
                    logger.error("   This can be caused by:")
                    logger.error("   - Missing or invalid mask file")
                    logger.error("   - GPU/CUDA not available")
                    logger.error("   - Invalid or corrupted AI model file")
                    logger.error("   - Insufficient GPU memory")
                    update_status_callback("Virhe: Flow estimatorin luonti epäonnistui")
                
                # Check for OpenCL errors
                if 'opencl init error' in stderr_lower:
                    logger.warning("⚠️  OpenCL initialization failed, falling back to CPU.")
                    logger.warning("   Stitching will be slower without OpenCL acceleration.")
                
                # Check for model path errors
                if 'model_path' in stderr_lower and 'error' in stderr_lower:
                    logger.error("❌ Error with AI model file path!")
                    logger.error(f"   Model path: {self.media_model_dir}/ai_stitcher_v2.ins")
                    logger.error("   Verify the path exists and the model file is valid.")
            
            logger.info(f"Videos stitched to {output_path}")
            update_status_callback("Stitchaus valmis")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Stitching failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
                
                stderr_lower = e.stderr.lower() if e.stderr else ""
                
                # Check if CUDA/GPU is not being used
                cuda_warnings = [
                    'cudacontext null',
                    'cuda context null',
                    'cuda not available',
                    'gpu not available',
                    'no cuda device',
                    'cuda initialization failed'
                ]
                
                cuda_not_used = any(warning in stderr_lower for warning in cuda_warnings)
                if cuda_not_used:
                    logger.warning("⚠️  CUDA/GPU is not being used for stitching! This may be causing the failure.")
                    logger.warning("   Check GPU drivers, CUDA installation, and MediaSDKTest configuration.")
                
                # Check for mask file error
                if 'can\'t find input: mask' in stderr_lower or 'can\'t find input mask' in stderr_lower:
                    logger.error("❌ Mask file is missing! This is likely causing the stitching failure.")
                    logger.error("   MediaSDKTest requires a mask file for AI stitching.")
                
                # Check for flow estimator errors
                if 'failed to create flow estimator' in stderr_lower:
                    logger.error("❌ Flow estimator creation failed! This is likely causing the stitching failure.")
                    logger.error("   Check mask file, GPU availability, and AI model file.")
            
            update_status_callback(f"Virhe stitchauksessa: {e.stderr or str(e)}")
            return False
        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            update_status_callback(f"Virhe stitchauksessa: {str(e)}")
            return False

