"""
Frame processing module for extracting, selecting, and processing video frames.
"""

import logging
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class FrameProcessing:
    """Handles frame extraction, selection, and processing."""
    
    def __init__(self, work_dir: Path, candidates_per_second: int, low_res_fps: int = 1, blur_settings: Optional[Dict] = None, source_fps: float = 29.95, target_fps: float = 10.0):
        """Initialize frame processing.
        
        Args:
            work_dir: Working directory for frame processing
            candidates_per_second: Number of candidate frames per second for selection
            low_res_fps: Target FPS for low-res frames
            blur_settings: Optional blur configuration
            source_fps: Source video FPS (default: 29.95)
            target_fps: Target FPS for extracted frames (default: 10.0)
        """
        self.work_dir = work_dir
        self.candidates_per_second = candidates_per_second
        self.low_res_fps = low_res_fps
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.blur_settings = blur_settings or {}
        self.blur_conf_threshold = float(self.blur_settings.get('confidence_threshold', 0.7))
        self.blur_min_box_area = float(self.blur_settings.get('min_box_area', 2500))
        self.blur_radius = float(self.blur_settings.get('blur_radius', 18))
        self._torch = None
        self._person_detector = None
        self._detector_device = None
        self._detector_transform = None
        self._init_person_detector()
    
    def _init_person_detector(self):
        """Lazy-load GPU person detector if torch/torchvision available."""
        if self.blur_settings.get('enable_gpu', True) is False:
            logger.info("GPU blur disabled via configuration")
            return
        try:
            import torch  # type: ignore
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights  # type: ignore
        except Exception as exc:
            logger.warning(f"GPU blur detector unavailable (missing torch/torchvision): {exc}")
            return
        
        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            model.to(device)
            model.eval()
            
            self._torch = torch
            self._person_detector = model
            self._detector_device = device
            self._detector_transform = weights.transforms()
            logger.info(f"Person detector initialized on device: {device}")
        except Exception as exc:
            logger.warning(f"Failed to initialize GPU blur detector: {exc}")
            self._person_detector = None
    
    def _extract_suffix(self, frame_path: Path) -> Optional[int]:
        """Extract numeric suffix from frame filename."""
        try:
            parts = frame_path.stem.split("_")
            if not parts:
                return None
            return int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse frame suffix from {frame_path}")
            return None
    
    def _calculate_sharpness(self, image_path: Path) -> float:
        """Calculate Laplacian variance as sharpness metric (optimized with downscaling)."""
        try:
            import cv2
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image for sharpness calculation: {image_path}")
                return 0.0
            
            # Downscale for faster calculation (maintains sharpness metric accuracy)
            # Target size: max 640x480 for ~4x speedup while maintaining accuracy
            height, width = img.shape
            if height > 480 or width > 640:
                scale = min(480 / height, 640 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Error calculating sharpness for {image_path}: {e}")
            return 0.0
    
    def extract_frames_high_and_low(self, video_path: Path, high_dir: Path, low_dir: Path, fps: float) -> Tuple[List[Path], List[Path]]:
        """Extract high and low resolution frames simultaneously."""
        logger.info(f"Extracting {fps} FPS frames (high and low res) from {video_path}...")
        
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return [], []
        
        if video_path.stat().st_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            return [], []
        
        high_dir.mkdir(parents=True, exist_ok=True)
        low_dir.mkdir(parents=True, exist_ok=True)
        
        try:
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
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            return [], []
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", exc_info=True)
            return [], []
    
    def _select_best_frames_by_sharpness(self, high_frames: List[Path], candidates_per_second: int) -> set[int]:
        """Select best frames based on sharpness from each group of candidates_per_second frames.
        
        Uses parallel processing to calculate sharpness for all frames simultaneously.
        """
        if not high_frames:
            return set()
        
        # Calculate sharpness in parallel for all high-res frames
        candidates_with_sharpness = []
        
        def calculate_sharpness_for_frame(frame_path: Path) -> Optional[Tuple[int, Path, float]]:
            """Calculate sharpness for a single frame."""
            try:
                # Extract frame number from filename (e.g., "high_000001.jpg" -> 1)
                suffix = int(frame_path.stem.split("_")[-1])
                sharpness = self._calculate_sharpness(frame_path)
                return (suffix, frame_path, sharpness)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse frame number from {frame_path}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel sharpness calculation
        # Use 8 workers for good balance between speed and resource usage
        logger.info(f"Calculating sharpness for {len(high_frames)} frames in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_frame = {
                executor.submit(calculate_sharpness_for_frame, frame_path): frame_path
                for frame_path in high_frames
            }
            
            completed = 0
            for future in as_completed(future_to_frame):
                result = future.result()
                if result:
                    candidates_with_sharpness.append(result)
                completed += 1
                # Log progress every 100 frames
                if completed % 100 == 0:
                    logger.debug(f"Sharpness calculation progress: {completed}/{len(high_frames)}")
        
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
    
    def _detect_people_boxes(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Run GPU-accelerated detection on low-res frame to find person bounding boxes."""
        if not self._person_detector or not self._torch or not self._detector_transform:
            return []
        
        try:
            transformed = self._detector_transform(image).to(self._detector_device)
            with self._torch.inference_mode():
                outputs = self._person_detector([transformed])[0]
            boxes = outputs.get('boxes', [])
            scores = outputs.get('scores', [])
            labels = outputs.get('labels', [])
            detected: List[Tuple[int, int, int, int]] = []
            for box, score, label in zip(boxes, scores, labels):
                if label.item() != 1:  # COCO class 1 = person
                    continue
                if score.item() < self.blur_conf_threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width * height < self.blur_min_box_area:
                    continue
                detected.append((int(x1), int(y1), int(x2), int(y2)))
            return detected
        except Exception as exc:
            logger.warning(f"GPU detection failed, falling back to no blur: {exc}")
            return []
    
    def _apply_blur_to_boxes(self, image_path: Path, boxes: List[Tuple[int, int, int, int]]) -> Optional[Path]:
        """Blur only the specified bounding boxes inside an image."""
        if not boxes:
            return None
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                for box in boxes:
                    x1, y1, x2, y2 = box
                    region = img.crop((x1, y1, x2, y2))
                    blurred = region.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
                    img.paste(blurred, (x1, y1, x2, y2))
                
                blurred_path = image_path.parent / f"b_{image_path.name}"
                img.save(blurred_path, quality=90)
                return blurred_path
        except Exception as exc:
            logger.error(f"Failed to blur image {image_path}: {exc}")
            return None
    
    def _apply_full_blur(self, image_path: Path) -> Optional[Path]:
        """Blur the entire image."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
                blurred_path = image_path.parent / f"b_{image_path.name}"
                blurred.save(blurred_path, quality=90)
                return blurred_path
        except Exception as exc:
            logger.error(f"Failed to apply full blur to image {image_path}: {exc}")
            return None
    
    def _scale_boxes(self, boxes: List[Tuple[int, int, int, int]], scale_x: float, scale_y: float) -> List[Tuple[int, int, int, int]]:
        """Scale bounding boxes according to provided multipliers."""
        scaled: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in boxes:
            scaled.append((
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ))
        return scaled
    
    def process_mediasdk_frames(self, frames_dir: Path, frame_indices: List[int], update_status_callback) -> Tuple[List[Path], List[Path]]:
        """
        Process frames extracted directly by MediaSDK.
        
        MediaSDK extracts high-resolution frames. This method:
        1. Loads the high-res frames
        2. Creates low-resolution versions (3840x1920)
        3. Maps frames to original frame indices for correct time calculation
        4. Returns both high and low res frame lists
        
        Args:
            frames_dir: Directory containing MediaSDK-extracted frames
            frame_indices: List of original frame indices (e.g., [0, 3, 6, 9, ...])
            update_status_callback: Status update callback
        
        Returns:
            Tuple of (high_res_frames, low_res_frames) lists
        """
        update_status_callback("Processing MediaSDK-extracted frames...")
        
        # Find all JPG frames in the directory
        high_frames = sorted(frames_dir.glob("*.jpg"))
        if not high_frames:
            logger.error(f"No frames found in {frames_dir}")
            update_status_callback("Error: No frames found")
            return [], []
        
        if len(high_frames) != len(frame_indices):
            logger.warning(f"Frame count mismatch: found {len(high_frames)} frames but expected {len(frame_indices)} indices")
            # Use available frames, truncate indices if needed
            frame_indices = frame_indices[:len(high_frames)]
        
        logger.info(f"Found {len(high_frames)} high-res frames from MediaSDK")
        update_status_callback(f"Processing {len(high_frames)} frames...")
        
        # Create low-res directory
        low_dir = self.work_dir / "low_frames"
        low_dir.mkdir(parents=True, exist_ok=True)
        
        # Create low-res versions, using original frame indices for naming
        low_frames = []
        for idx, (high_frame, original_frame_idx) in enumerate(zip(high_frames, frame_indices)):
            try:
                # Use original frame index for naming (e.g., frame 0, 3, 6, 9...)
                frame_num = original_frame_idx
                
                # Create low-res version
                low_frame_path = low_dir / f"low_{frame_num:06d}.jpg"
                
                with Image.open(high_frame) as img:
                    # Resize to low resolution (3840x1920)
                    low_img = img.resize((3840, 1920), Image.Resampling.LANCZOS)
                    low_img.save(low_frame_path, quality=90)
                
                low_frames.append(low_frame_path)
                
                if (idx + 1) % 10 == 0:
                    update_status_callback(f"Processed {idx + 1}/{len(high_frames)} frames...")
                    
            except Exception as e:
                logger.warning(f"Failed to process frame {high_frame}: {e}")
                continue
        
        # Move high frames to selected directory for consistency
        selected_dir = self.work_dir / "selected_frames"
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        final_high = []
        for idx, (high_frame, original_frame_idx) in enumerate(zip(high_frames, frame_indices)):
            try:
                frame_num = original_frame_idx
                new_high = selected_dir / f"high_{frame_num:06d}.jpg"
                shutil.move(str(high_frame), str(new_high))
                final_high.append(new_high)
            except Exception as e:
                logger.warning(f"Failed to move high frame {high_frame}: {e}")
        
        # Move low frames to selected directory
        final_low = []
        for low_frame in low_frames:
            try:
                frame_num = int(low_frame.stem.split("_")[-1])
                new_low = selected_dir / f"low_{frame_num:06d}.jpg"
                shutil.move(str(low_frame), str(new_low))
                final_low.append(new_low)
            except Exception as e:
                logger.warning(f"Failed to move low frame {low_frame}: {e}")
        
        # Sort by frame number (original frame index)
        final_high.sort(key=lambda f: int(f.stem.split("_")[-1]))
        final_low.sort(key=lambda f: int(f.stem.split("_")[-1]))
        
        logger.info(f"Processed {len(final_high)} high-res and {len(final_low)} low-res frames")
        update_status_callback(f"Processed {len(final_high)} frame pairs")
        return final_high, final_low
    
    def create_and_select_frames(self, stitched_path: Path, update_status_callback) -> Tuple[List[Path], List[Path]]:
        """Extract 1fps frames (high and low res) for API storage.
        
        NOTE: This method is kept for backward compatibility.
        Use process_mediasdk_frames() when frames are extracted directly by MediaSDK.
        """
        update_status_callback(f"Creating frames ({self.low_res_fps}/s) from stitched file...")
        
        high_dir = self.work_dir / "high_frames"
        low_dir = self.work_dir / "low_frames"
        
        # Extract high and low res frames at 1 fps
        # For 1 fps, we still extract at 12 fps first to select sharpest, then keep only 1 per second
        high_frames, low_frames = self.extract_frames_high_and_low(
            stitched_path, high_dir, low_dir, self.candidates_per_second
        )
        
        if not high_frames or not low_frames:
            update_status_callback("Error: Frame creation failed")
            return [], []
        
        update_status_callback(f"Created {len(high_frames)} high res and {len(low_frames)} low res frames")
        
        # Select best frames based on sharpness (one per second)
        update_status_callback("Selecting sharpest frames from each second...")
        selected_suffixes = self._select_best_frames_by_sharpness(high_frames, self.candidates_per_second)
        
        # Filter frames to only selected ones
        selected_high = [f for f in high_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        selected_low = [f for f in low_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        
        # Create sets for faster lookup
        selected_high_set = set(selected_high)
        selected_low_set = set(selected_low)
        
        # Batch delete unselected frames in parallel
        unselected_high = [f for f in high_frames if f not in selected_high_set]
        unselected_low = [f for f in low_frames if f not in selected_low_set]
        
        if unselected_high or unselected_low:
            def safe_delete(path: Path):
                """Safely delete a file."""
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.debug(f"Failed to delete {path}: {e}")
            
            # Delete unselected frames in parallel
            logger.debug(f"Deleting {len(unselected_high) + len(unselected_low)} unselected frames...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                for frame in unselected_high + unselected_low:
                    executor.submit(safe_delete, frame)
        
        # Move selected frames to final directory
        selected_dir = self.work_dir / "selected_frames"
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        final_high = []
        final_low = []
        
        # Sort selected frames by suffix to ensure correct pairing
        selected_high.sort(key=lambda f: int(f.stem.split("_")[-1]))
        selected_low.sort(key=lambda f: int(f.stem.split("_")[-1]))
        
        # Move frames in parallel
        def move_frame_pair(high_frame: Path, low_frame: Path) -> Tuple[Path, Path]:
            """Move a pair of frames to selected directory."""
            suffix = high_frame.stem.split("_")[-1]
            new_high = selected_dir / f"high_{suffix}.jpg"
            new_low = selected_dir / f"low_{suffix}.jpg"
            
            shutil.move(str(high_frame), str(new_high))
            shutil.move(str(low_frame), str(new_low))
            
            return new_high, new_low
        
        logger.debug(f"Moving {len(selected_high)} selected frame pairs...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pair = {
                executor.submit(move_frame_pair, high, low): (high, low)
                for high, low in zip(selected_high, selected_low)
            }
            
            for future in as_completed(future_to_pair):
                new_high, new_low = future.result()
                final_high.append(new_high)
                final_low.append(new_low)
        
        # Sort final lists by suffix
        final_high.sort(key=lambda f: int(f.stem.split("_")[-1]))
        final_low.sort(key=lambda f: int(f.stem.split("_")[-1]))
        
        update_status_callback(f"Selected {len(final_high)} best frames ({self.low_res_fps}/s)")
        return final_high, final_low
    
    def blur_frames_optional(self, high_frames: List[Path], low_frames: List[Path], blur_people: bool, update_status_callback) -> Tuple[List[Path], List[Path]]:
        """Optionally blur frames if blur_people is enabled."""
        if not blur_people:
            logger.info("Blur not enabled, skipping blur step")
            return high_frames, low_frames
        
        logger.info("Applying GPU-assisted blur to frames...")
        update_status_callback("Detecting people and applying blur...")
        
        detector_available = self._person_detector is not None
        if not detector_available and not self.blur_settings.get('fallback_full_frame', True):
            logger.warning("Person detector unavailable and fallback disabled; returning original frames")
            return high_frames, low_frames
        
        blurred_high: List[Path] = []
        blurred_low: List[Path] = []
        
        try:
            for high_frame, low_frame in zip(high_frames, low_frames):
                boxes: List[Tuple[int, int, int, int]] = []
                low_size = None
                high_size = None
                
                try:
                    with Image.open(low_frame) as low_img:
                        low_img = low_img.convert("RGB")
                        low_size = low_img.size
                        if detector_available:
                            boxes = self._detect_people_boxes(low_img)
                except Exception as exc:
                    logger.warning(f"Failed to process low frame {low_frame}: {exc}")
                
                if boxes:
                    blurred_low_path = self._apply_blur_to_boxes(low_frame, boxes) or low_frame
                elif self.blur_settings.get('fallback_full_frame', True):
                    blurred_low_path = self._apply_full_blur(low_frame) or low_frame
                else:
                    blurred_low_path = low_frame
                
                try:
                    with Image.open(high_frame) as high_img:
                        high_img = high_img.convert("RGB")
                        high_size = high_img.size
                except Exception as exc:
                    logger.warning(f"Failed to open high frame {high_frame}: {exc}")
                
                if boxes and low_size and high_size:
                    scale_x = high_size[0] / low_size[0]
                    scale_y = high_size[1] / low_size[1]
                    scaled_boxes = self._scale_boxes(boxes, scale_x, scale_y)
                    blurred_high_path = self._apply_blur_to_boxes(high_frame, scaled_boxes) or high_frame
                elif self.blur_settings.get('fallback_full_frame', True):
                    blurred_high_path = self._apply_full_blur(high_frame) or high_frame
                else:
                    blurred_high_path = high_frame
                
                blurred_high.append(blurred_high_path)
                blurred_low.append(blurred_low_path)
            
            logger.info(f"Blurred {len(blurred_high)} frame pairs")
            update_status_callback(f"Blurred {len(blurred_high)} frames")
            return blurred_high, blurred_low
        except Exception as e:
            logger.error(f"Frame blurring failed: {e}")
            update_status_callback(f"Error in frame blurring: {str(e)}")
            return high_frames, low_frames
    
    def upload_frames_to_cloud(self, frame_paths: List[Path], project_id: str, video_id: Optional[str], api_client, update_status_callback, layer_id: Optional[str] = None, max_workers: int = 5) -> List[Dict]:
        """Save frames to cloud and create frame objects using parallel uploads.
        
        Args:
            frame_paths: List of frame paths to upload
            project_id: Project ID
            video_id: Video ID
            api_client: API client instance
            update_status_callback: Status update callback
            layer_id: Optional layer ID
            max_workers: Number of parallel upload workers (default: 5)
        """
        logger.info("Uploading frames to cloud...")
        update_status_callback(f"Saving {len(frame_paths)} frames to cloud and creating frame objects...")

        if not video_id:
            logger.error("Cannot upload frames: missing target video_id")
            update_status_callback("Error: video_id missing, cannot save frames")
            return []
        
        frame_objects = []
        total = len(frame_paths)
        frame_collections: Dict[int, Dict[str, Optional[str]]] = defaultdict(lambda: {
            'high': None,
            'low': None,
            'blurred_high': None,
            'blurred_low': None
        })
        
        def upload_single_frame(frame_path: Path, index: int) -> Optional[Dict]:
            """Upload a single frame and return frame object."""
            try:
                # Read image binary
                with open(frame_path, 'rb') as f:
                    image_binary = f.read()
                
                # Determine image type from filename
                image_type = 'high' if 'high' in frame_path.name else 'low'
                if 'b_' in frame_path.name:
                    image_type = f'blurred_{image_type}'
                
                # Store image using new store_image function
                image_id = api_client.store_image(
                    project_id=project_id,
                    image_type=image_type,
                    image_size=len(image_binary),
                    image_binary=image_binary
                )
                
                if image_id:
                    frame_obj = {
                        'uuid': image_id,
                        'type': image_type,
                        'path': str(frame_path),
                        'index': index,
                        'suffix': self._extract_suffix(frame_path)
                    }
                    return frame_obj
                else:
                    logger.warning(f"Failed to store frame {index}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to upload frame {index}: {e}")
                return None
        
        # Upload frames in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_index = {
                executor.submit(upload_single_frame, frame_path, i): i
                for i, frame_path in enumerate(frame_paths)
            }
            
            # Process completed uploads as they finish
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    frame_obj = future.result()
                    if frame_obj:
                        frame_objects.append(frame_obj)
                        
                        # Update frame collections
                        suffix = frame_obj.get('suffix')
                        image_type = frame_obj.get('type')
                        if suffix is not None:
                            frame_collections[suffix][image_type] = frame_obj['uuid']
                            # Log in test mode
                            if hasattr(api_client, 'test_mode') and api_client.test_mode:
                                print(f"  Stored {image_type} image for suffix {suffix}: {frame_obj['uuid']}")
                        
                        completed += 1
                        # Update progress every 10 frames or at the end
                        if completed % 10 == 0 or completed == total:
                            update_status_callback(f"Saving frames to cloud: {completed}/{total}")
                except Exception as e:
                    logger.error(f"Error processing upload result for frame {index}: {e}")
        
        # Sort frame_objects by original index to maintain order
        frame_objects.sort(key=lambda x: x.get('index', 0))
        # Remove index from final objects
        for obj in frame_objects:
            obj.pop('index', None)
        
        logger.info(f"Uploaded {len(frame_objects)} frames")
        update_status_callback(f"Saved {len(frame_objects)} frames to cloud")
        
        # In test mode, print summary of frame collections
        if hasattr(api_client, 'test_mode') and api_client.test_mode:
            print(f"\n=== FRAME COLLECTIONS SUMMARY ===")
            for suffix, images in sorted(frame_collections.items()):
                if any(images.values()):  # Only print if there are any images
                    print(f"  Suffix {suffix}: {images}")
            print("=" * 60)
        
        if video_id:
            video_frame_ids = self.store_video_frames(
                frame_collections,
                project_id,
                video_id,
                api_client,
                update_status_callback,
                layer_id=layer_id
            )
            logger.info(f"Created {len(video_frame_ids)} video frame records")
        
        return frame_objects
    
    def store_video_frames(self, frame_collections: Dict[int, Dict[str, Optional[str]]], project_id: str, video_id: str, api_client, update_status_callback, layer_id: Optional[str] = None) -> List[str]:
        """Create video frame objects in API using uploaded image references."""
        saved_frame_ids: List[str] = []
        if not frame_collections:
            return saved_frame_ids
        
        sorted_suffixes = sorted(frame_collections.keys())
        update_status_callback("Saving video frame objects...")
        
        frame_entries: List[Dict] = []
        total = len(sorted_suffixes)
        for idx, suffix in enumerate(sorted_suffixes):
            images = frame_collections[suffix]
            high_id = images.get('high') or images.get('blurred_high')
            low_id = images.get('low') or images.get('blurred_low')
            blur_high_id = images.get('blurred_high')
            blur_low_id = images.get('blurred_low')
            
            if not high_id or not low_id:
                logger.warning(f"Skipping suffix {suffix}: missing high/low images")
                logger.warning(f"  Available images: {images}")
                continue
            
            # Calculate time_in_video based on source frame index and target FPS
            # suffix is the original frame index (e.g., 0, 3, 6, 9...)
            # time = original_frame_index / source_fps
            # But we want to map to target FPS, so: time = original_frame_index / source_fps
            # However, since we extract every Nth frame, we can also use: time = (suffix / source_fps)
            # For 10 fps from 29.95 fps (every 3rd frame), frame 3 = 3/29.95 = 0.1s
            time_in_video = suffix / float(self.source_fps)
            
            if hasattr(api_client, 'test_mode') and api_client.test_mode:
                print(f"\n--- Preparing video-frame payload for suffix {suffix} (time: {time_in_video}s) ---")
                print(f"  high_id: {high_id}")
                print(f"  low_id: {low_id}")
                print(f"  blur_high_id: {blur_high_id}")
                print(f"  blur_low_id: {blur_low_id}")
                print(f"  layer_id: {layer_id}")
                print(f"  All images in collection: {images}")
            
            payload = {
                'project': project_id,
                'video': video_id,
                'time_in_video': int(time_in_video),
                'high_image': high_id,
                'image': low_id
            }
            
            if blur_high_id:
                payload['blur_high_image'] = blur_high_id
            if blur_low_id:
                payload['blur_image'] = blur_low_id
            if layer_id:
                payload['camera_layer_position'] = {
                    "rx": 0,
                    "ry": 0,
                    "rz": 0,
                    "layer": layer_id
                }
            
            frame_entries.append({
                'suffix': suffix,
                'time_in_video': time_in_video,
                'high_id': high_id,
                'low_id': low_id,
                'blur_high_id': blur_high_id,
                'blur_low_id': blur_low_id,
                'payload': payload
            })
            
            if (idx + 1) % 5 == 0 or idx == total - 1:
                update_status_callback(f"Preparing video frames: {idx + 1}/{total}")
        
        if not frame_entries:
            update_status_callback("No frames to save to API")
            return saved_frame_ids
        
        bulk_payloads = [entry['payload'] for entry in frame_entries]
        update_status_callback(f"Sending {len(bulk_payloads)} video frames in bulk request...")
        bulk_response = api_client.save_video_frames_bulk(bulk_payloads)
        
        if bulk_response is None:
            logger.warning("Bulk video frame luonti epäonnistui, yritetään yksittäisin kutsuin")
            fallback_ids = self._store_video_frames_individual(frame_entries, api_client, update_status_callback, layer_id)
            saved_frame_ids.extend(fallback_ids)
            update_status_callback(f"Saved {len(saved_frame_ids)} video frames to API (fallback)")
            return saved_frame_ids
        
        failed_entries: List[Dict] = []
        for idx, entry in enumerate(frame_entries):
            response_entry = bulk_response[idx] if idx < len(bulk_response) else None
            status_code = response_entry.get('status_code') if response_entry else None
            frame_uuid = response_entry.get('uuid') if response_entry else None
            
            if status_code in (200, 201) and frame_uuid:
                saved_frame_ids.append(frame_uuid)
            else:
                failed_entries.append(entry)
                error_msg = (response_entry or {}).get('error', 'Tuntematon virhe')
                logger.warning(f"Video framen tallennus epäonnistui suffixille {entry['suffix']}: {error_msg}")
        
        if failed_entries:
            logger.warning(f"{len(failed_entries)} video framea epäonnistui bulk-vastauksessa, yritetään yksittäin...")
            update_status_callback(f"Bulk tallennus epäonnistui {len(failed_entries)} framella, yritetään yksitellen...")
            fallback_ids = self._store_video_frames_individual(failed_entries, api_client, update_status_callback, layer_id)
            saved_frame_ids.extend(fallback_ids)
        
        update_status_callback(f"Saved {len(saved_frame_ids)} video frames to API")
        return saved_frame_ids
    
    def _store_video_frames_individual(self, frame_entries: List[Dict], api_client, update_status_callback, layer_id: Optional[str]) -> List[str]:
        """Fallback to storing video frames one by one if bulk call fails."""
        saved_frame_ids: List[str] = []
        total = len(frame_entries)
        
        for idx, entry in enumerate(frame_entries):
            frame_uuid = api_client.save_videoframe(
                project_id=entry['payload']['project'],
                video_id=entry['payload']['video'],
                time_in_video=entry['time_in_video'],
                high_res_image_id=entry['high_id'],
                low_res_image_id=entry['low_id'],
                blur_high_image_id=entry['blur_high_id'],
                blur_low_image_id=entry['blur_low_id'],
                layer_id=layer_id
            )
            
            if frame_uuid:
                saved_frame_ids.append(frame_uuid)
                if (idx + 1) % 5 == 0 or idx == total - 1:
                    update_status_callback(f"Individual save: {idx + 1}/{total}")
            else:
                logger.warning(f"Failed to create video frame (fallback) for suffix {entry['suffix']}")
        
        return saved_frame_ids
    
    def create_stella_frames(self, stitched_path: Path, update_status_callback) -> List[Path]:
        """
        Create 12fps frames for Stella VSLAM route calculation.
        These are separate from the selected frames used for API storage.
        
        Args:
            stitched_path: Path to stitched video
            update_status_callback: Status update callback
            
        Returns:
            List of frame paths for Stella
        """
        update_status_callback("Creating 12fps frames for Stella route calculation...")
        
        stella_dir = self.work_dir / "stella_frames"
        stella_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract 12fps frames at low resolution (3840x1920) for Stella
        # Use candidates_per_second (12) for frame rate
        stella_pattern = str(stella_dir / "stella_%06d.jpg")
        stella_cmd = [
            'ffmpeg',
            '-i', str(stitched_path),
            '-vf', f'fps={self.candidates_per_second}/1:round=up',
            '-s', '3840x1920',  # Low resolution for Stella
            '-q:v', '0',  # Max quality
            '-start_number', '0',
            stella_pattern
        ]
        
        try:
            result = subprocess.run(stella_cmd, check=True, capture_output=True, text=True)
            stella_frames = sorted(stella_dir.glob("stella_*.jpg"))
            logger.info(f"Created {len(stella_frames)} Stella frames at {self.candidates_per_second} fps")
            update_status_callback(f"Created {len(stella_frames)} Stella frames ({self.candidates_per_second}/s)")
            return stella_frames
        except subprocess.CalledProcessError as e:
            logger.error(f"Stella frame extraction failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            update_status_callback("Error: Stella frame creation failed")
            return []
        except Exception as e:
            logger.error(f"Stella frame extraction failed: {e}", exc_info=True)
            update_status_callback("Error: Stella frame creation failed")
            return []
    
    def get_route_frames_from_low_res(self, low_frames: List[Path], update_status_callback) -> List[Path]:
        """Use 12fps low res frames for route calculation."""
        logger.info(f"Using {len(low_frames)} low res frames for route calculation...")
        update_status_callback(f"Using {len(low_frames)} low res frames for route calculation...")
        return low_frames

