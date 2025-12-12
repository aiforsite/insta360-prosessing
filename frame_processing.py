"""
Frame processing module for extracting, selecting, and processing video frames.
"""

import logging
import re
import shutil
import subprocess
import requests
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
        
        Groups frames by second (based on frame number and candidates_per_second FPS),
        then selects the sharpest frame from each second.
        
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
        
        # Sort by suffix (frame number)
        candidates_with_sharpness.sort(key=lambda x: x[0])
        
        # Log first few and last few frame numbers for debugging
        if candidates_with_sharpness:
            first_few = [x[0] for x in candidates_with_sharpness[:5]]
            last_few = [x[0] for x in candidates_with_sharpness[-5:]]
            logger.info(f"Frame number range: first few={first_few}, last few={last_few}, total={len(candidates_with_sharpness)}")
        
        # Group frames by second: frames are at candidates_per_second FPS, so
        # frames 0-11 are in second 0, frames 12-23 are in second 1, etc.
        # Calculate which second each frame belongs to: second = frame_number // candidates_per_second
        frames_by_second: Dict[int, List[Tuple[int, Path, float]]] = {}
        for suffix, path, sharpness in candidates_with_sharpness:
            second = suffix // candidates_per_second
            if second not in frames_by_second:
                frames_by_second[second] = []
            frames_by_second[second].append((suffix, path, sharpness))
        
        # Select sharpest frame from each second
        selected_suffixes = set()
        total_seconds = len(frames_by_second)
        logger.info(f"Grouping {len(candidates_with_sharpness)} frames into {total_seconds} seconds (at {candidates_per_second} fps)")
        
        for second in sorted(frames_by_second.keys()):
            second_group = frames_by_second[second]
            if not second_group:
                continue
            
            # Select sharpest from this second
            best_suffix, best_path, best_sharpness = max(second_group, key=lambda x: x[2])
            selected_suffixes.add(best_suffix)
            
            group_frame_numbers = [x[0] for x in second_group]
            logger.debug(f"Second {second}: group frames={group_frame_numbers} (count={len(group_frame_numbers)}), selected frame {best_suffix} (sharpness={best_sharpness:.2f})")
        
        logger.info(f"Selected {len(selected_suffixes)} best frames from {len(candidates_with_sharpness)} candidates across {total_seconds} seconds")
        if len(selected_suffixes) != total_seconds:
            logger.warning(f"Expected {total_seconds} selected frames (one per second) but got {len(selected_suffixes)}")
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
    
    def process_mediasdk_frames(self, frames_dir: Path, frame_indices: List[int], update_status_callback) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Process frames extracted directly by MediaSDK.
        
        MediaSDK extracts 10 frames per second. This method:
        1. Loads all high-res frames
        2. Creates low-resolution versions (3840x1920) for all frames
        3. Selects the sharpest frame from each second (10 frames -> 1 frame)
        4. Returns selected frames for API storage (1 fps) and all low-res frames for Stella (10 fps)
        
        Args:
            frames_dir: Directory containing MediaSDK-extracted frames
            frame_indices: List of original frame indices (e.g., [0, 3, 6, 9, ...])
            update_status_callback: Status update callback
        
        Returns:
            Tuple of (selected_high_res, selected_low_res, all_low_res_stella)
            - selected_*: Best frames for API storage (1 fps)
            - all_low_res_stella: All low-res frames for Stella route calculation (10 fps)
        """
        update_status_callback("Processing MediaSDK-extracted frames...")
        
        # Find all JPG frames in the directory
        high_frames = sorted(frames_dir.glob("*.jpg"))
        if not high_frames:
            logger.error(f"No frames found in {frames_dir}")
            update_status_callback("Error: No frames found")
            return [], [], [], []
        
        if len(high_frames) != len(frame_indices):
            logger.warning(f"Frame count mismatch: found {len(high_frames)} frames but expected {len(frame_indices)} indices")
            # Use available frames, truncate indices if needed
            frame_indices = frame_indices[:len(high_frames)]
        
        logger.info(f"Found {len(high_frames)} high-res frames from MediaSDK (10 fps)")
        update_status_callback(f"Processing {len(high_frames)} frames...")
        
        # Create temporary directory for processing
        temp_dir = self.work_dir / "temp_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Move frames to temp directory with proper naming
        temp_high_frames = []
        for idx, (high_frame, original_frame_idx) in enumerate(zip(high_frames, frame_indices)):
            try:
                frame_num = original_frame_idx
                temp_high = temp_dir / f"high_{frame_num:06d}.jpg"
                shutil.move(str(high_frame), str(temp_high))
                temp_high_frames.append(temp_high)
            except Exception as e:
                logger.warning(f"Failed to move high frame {high_frame}: {e}")
                continue
        
        # Optimize MediaSDK frames by re-compressing with optimized JPG quality
        # MediaSDK creates unoptimized JPG files (~7MB), re-compressing reduces size significantly (~3MB)
        logger.info("Optimizing MediaSDK JPG frames...")
        update_status_callback("Optimizing JPG compression...")
        optimized_count = 0
        total_size_before = 0
        total_size_after = 0
        
        for temp_high in temp_high_frames:
            try:
                # Get original file size
                original_size = temp_high.stat().st_size
                total_size_before += original_size
                
                # Open and re-save with optimized quality
                with Image.open(temp_high) as img:
                    # Convert to RGB if necessary (some formats may have alpha channel)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save with quality=95 (high quality but optimized compression)
                    # optimize=True enables Huffman table optimization
                    temp_high_optimized = temp_high.parent / f"opt_{temp_high.name}"
                    img.save(temp_high_optimized, quality=95, optimize=True)
                    
                    # Replace original with optimized
                    optimized_size = temp_high_optimized.stat().st_size
                    total_size_after += optimized_size
                    
                    temp_high.unlink()
                    temp_high_optimized.rename(temp_high)
                    
                    optimized_count += 1
                    
                    # Log progress every 10 frames
                    if optimized_count % 10 == 0:
                        reduction = ((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0
                        logger.debug(f"Optimized {optimized_count}/{len(temp_high_frames)} frames (avg reduction: {reduction:.1f}%)")
                        
            except Exception as e:
                logger.warning(f"Failed to optimize MediaSDK frame {temp_high}: {e}")
                continue
        
        if optimized_count > 0:
            total_reduction = ((total_size_before - total_size_after) / total_size_before * 100) if total_size_before > 0 else 0
            logger.info(f"Optimized {optimized_count}/{len(temp_high_frames)} MediaSDK frames: "
                       f"{total_size_before / (1024*1024):.2f}MB -> {total_size_after / (1024*1024):.2f}MB "
                       f"({total_reduction:.1f}% reduction)")
            update_status_callback(f"Optimized {optimized_count} frames ({total_reduction:.1f}% size reduction)")
        
        # Create low-res versions for all frames
        low_dir = self.work_dir / "low_frames"
        low_dir.mkdir(parents=True, exist_ok=True)
        
        all_low_frames = []
        for temp_high in temp_high_frames:
            try:
                frame_num = int(temp_high.stem.split("_")[-1])
                low_frame_path = low_dir / f"low_{frame_num:06d}.jpg"
                
                with Image.open(temp_high) as img:
                    # Resize to low resolution (3840x1920)
                    low_img = img.resize((3840, 1920), Image.Resampling.LANCZOS)
                    low_img.save(low_frame_path, quality=90)
                
                all_low_frames.append(low_frame_path)
                
            except Exception as e:
                logger.warning(f"Failed to create low-res frame for {temp_high}: {e}")
                continue
        
        # Sort by frame number
        temp_high_frames.sort(key=lambda f: int(f.stem.split("_")[-1]))
        all_low_frames.sort(key=lambda f: int(f.stem.split("_")[-1]))
        
        logger.info(f"Created {len(all_low_frames)} low-res frames")
        update_status_callback(f"Created low-res versions for {len(all_low_frames)} frames...")
        
        # Select best frame from each second (10 frames -> 1 frame)
        # We have 10 frames per second, so candidates_per_second should be 10
        frames_per_second = int(self.target_fps)  # 10 frames per second
        update_status_callback("Selecting sharpest frame from each second...")
        selected_suffixes = self._select_best_frames_by_sharpness(temp_high_frames, frames_per_second)
        
        # Filter to only selected frames
        selected_high = [f for f in temp_high_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        selected_low = [f for f in all_low_frames if int(f.stem.split("_")[-1]) in selected_suffixes]
        
        # Move all low-res frames to Stella directory (for route calculation)
        # Stella only needs low-res frames (3840x1920), not high-res
        stella_dir = self.work_dir / "stella_frames"
        stella_dir.mkdir(parents=True, exist_ok=True)
        
        stella_low = []
        
        # Move all low frames to Stella directory
        # IMPORTANT: Stella uses frame filename as timestamp, so we need to name frames using sequential numbering
        # Frame indices are original video frame numbers (0, 3, 6, 9...), but Stella needs sequential numbering
        # representing time in seconds at 10 fps (0.0s, 0.1s, 0.2s...)
        # So frame index 0 = 0.0s = stella_000000.jpg, frame index 3 = 0.1s = stella_000001.jpg, etc.
        logger.info(f"Naming {len(all_low_frames)} frames for Stella using sequential numbering (Stella uses frame number as timestamp)")
        for idx, low_frame in enumerate(all_low_frames):
            try:
                # Get original frame index from filename
                original_frame_idx = int(low_frame.stem.split("_")[-1])
                
                # Use sequential index (0, 1, 2, 3...) representing frame order at 10 fps
                # This will be interpreted by Stella as timestamps: 0.0s, 0.1s, 0.2s, 0.3s...
                stella_frame_num = idx  # Sequential frame number (0, 1, 2, 3...)
                stella_low_path = stella_dir / f"stella_{stella_frame_num:06d}.jpg"
                
                # Log first few and last few mappings
                if idx < 5 or idx >= len(all_low_frames) - 5:
                    logger.info(f"Stella frame {stella_frame_num}: original_index={original_frame_idx}, "
                               f"expected_time={stella_frame_num * 0.1:.1f}s, filename={stella_low_path.name}")
                
                shutil.move(str(low_frame), str(stella_low_path))
                stella_low.append(stella_low_path)
            except Exception as e:
                logger.warning(f"Failed to move Stella low frame {low_frame}: {e}")
        
        # Move high frames to selected directory (they'll be used for API if selected)
        # High frames not selected will be deleted later
        high_dir = self.work_dir / "high_frames"
        high_dir.mkdir(parents=True, exist_ok=True)
        
        for high_frame in temp_high_frames:
            try:
                frame_num = int(high_frame.stem.split("_")[-1])
                high_path = high_dir / f"high_{frame_num:06d}.jpg"
                shutil.move(str(high_frame), str(high_path))
            except Exception as e:
                logger.warning(f"Failed to move high frame {high_frame}: {e}")
        
        # Copy selected frames to selected directory (for API storage)
        selected_dir = self.work_dir / "selected_frames"
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        final_selected_high = []
        final_selected_low = []
        
        # Create mapping from original frame index to sequential index for Stella frames
        # Stella frames are named using sequential index (0, 1, 2, 3...) representing time at 10 fps
        frame_index_to_stella_index = {orig_idx: seq_idx for seq_idx, orig_idx in enumerate(frame_indices)}
        
        # Find selected frames and copy them to selected directory
        for frame_num in selected_suffixes:
            try:
                # Find high frame in high_dir (uses original frame index)
                high_path = high_dir / f"high_{frame_num:06d}.jpg"
                if high_path.exists():
                    selected_high_path = selected_dir / f"high_{frame_num:06d}.jpg"
                    shutil.copy2(str(high_path), str(selected_high_path))
                    final_selected_high.append(selected_high_path)
                
                # Find low frame in Stella directory (uses sequential index, not original frame index)
                stella_frame_num = frame_index_to_stella_index.get(frame_num)
                if stella_frame_num is not None:
                    stella_low_path = stella_dir / f"stella_{stella_frame_num:06d}.jpg"
                    if stella_low_path.exists():
                        selected_low_path = selected_dir / f"low_{frame_num:06d}.jpg"
                        shutil.copy2(str(stella_low_path), str(selected_low_path))
                        final_selected_low.append(selected_low_path)
            except Exception as e:
                logger.warning(f"Failed to copy selected frame {frame_num} to selected directory: {e}")
        
        # Delete unselected high frames to save space
        for high_path in high_dir.glob("high_*.jpg"):
            try:
                frame_num = int(high_path.stem.split("_")[-1])
                if frame_num not in selected_suffixes:
                    high_path.unlink()
            except Exception as e:
                logger.debug(f"Failed to delete unselected high frame {high_path}: {e}")
        
        # Sort all lists
        final_selected_high.sort(key=lambda f: int(f.stem.split("_")[-1]))
        final_selected_low.sort(key=lambda f: int(f.stem.split("_")[-1]))
        stella_low.sort(key=lambda f: int(f.stem.split("_")[-1]))
        
        logger.info(f"Selected {len(final_selected_high)} best frames (1 fps) for API storage")
        logger.info(f"Prepared {len(stella_low)} low-res frames (10 fps) for Stella route calculation")
        update_status_callback(f"Selected {len(final_selected_high)} best frames (1 fps) for API, {len(stella_low)} low-res frames for Stella")
        
        return final_selected_high, final_selected_low, stella_low
    
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
        """Save frames to cloud and create frame objects using batch-create endpoint.
        
        New flow:
        1. Group frames by suffix (high/low/blurred)
        2. Calculate times_in_video for each frame
        3. Call batch-create to create frames and images (gets upload URLs)
        4. Upload image binaries to upload URLs
        
        Args:
            frame_paths: List of frame paths to upload
            project_id: Project ID
            video_id: Video ID
            api_client: API client instance
            update_status_callback: Status update callback
            layer_id: Optional layer ID
            max_workers: Number of parallel upload workers (default: 5)
        """
        logger.info("Uploading frames to cloud using batch-create...")
        update_status_callback(f"Saving {len(frame_paths)} frames to cloud and creating frame objects...")

        if not video_id:
            logger.error("Cannot upload frames: missing target video_id")
            update_status_callback("Error: video_id missing, cannot save frames")
            return []
        
        if not layer_id:
            logger.error("Cannot upload frames: missing layer_id")
            update_status_callback("Error: layer_id missing, cannot save frames")
            return []
        
        # Group frames by suffix and type
        frames_by_suffix: Dict[int, Dict[str, Path]] = defaultdict(dict)
        for frame_path in frame_paths:
            suffix = self._extract_suffix(frame_path)
            if suffix is None:
                logger.warning(f"Could not extract suffix from {frame_path.name}, skipping")
                continue
            
            # Determine image type from filename
            if 'high' in frame_path.name:
                if 'b_' in frame_path.name:
                    frames_by_suffix[suffix]['blurred_high'] = frame_path
                else:
                    frames_by_suffix[suffix]['high'] = frame_path
            else:  # low
                if 'b_' in frame_path.name:
                    frames_by_suffix[suffix]['blurred_low'] = frame_path
                else:
                    frames_by_suffix[suffix]['low'] = frame_path
        
        if not frames_by_suffix:
            logger.error("No valid frames found to upload")
            return []
        
        # Calculate times_in_video for batch-create
        # Framet on poimittu candidates_per_second fps:llä (12 fps) ja valittu 1 fps:llä (yksi per sekunti)
        # Joten time_in_video lasketaan: suffix / candidates_per_second
        # Esim. frame 0 -> 0/12 = 0.0s, frame 12 -> 12/12 = 1.0s, frame 24 -> 24/12 = 2.0s
        sorted_suffixes = sorted(frames_by_suffix.keys())
        times_in_video = [suffix / float(self.candidates_per_second) for suffix in sorted_suffixes]
        
        # Check if blur images exist
        blur_people = any(
            'blurred_high' in frames_by_suffix[suffix] or 'blurred_low' in frames_by_suffix[suffix]
            for suffix in sorted_suffixes
        )
        
        # Step 1: Create frames and images using batch-create
        update_status_callback(f"Creating {len(times_in_video)} video frames with batch-create...")
        created_frames = api_client.batch_create_video_frames(
            video_id=video_id,
            times_in_video=times_in_video,
            layer_id=layer_id,
            blur_people=blur_people
        )
        
        if not created_frames:
            logger.error("Batch-create video frames failed")
            return []
        
        logger.info(f"Created {len(created_frames)} video frames with batch-create")
        # Log first frame structure for debugging
        if created_frames and len(created_frames) > 0:
            logger.debug(f"First created frame structure: {created_frames[0]}")
        
        # Step 2: Upload image binaries to upload URLs
        update_status_callback(f"Uploading {len(frame_paths)} image binaries...")
        
        def upload_image_binary(upload_url: str, image_path: Path) -> bool:
            """Upload image binary to upload URL."""
            try:
                with open(image_path, 'rb') as f:
                    image_binary = f.read()
                
                response = requests.put(
                    upload_url,
                    data=image_binary,
                    headers={'Content-Type': 'application/octet-stream'},
                    timeout=300
                )
                response.raise_for_status()
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to upload image {image_path} to {upload_url}: {e}")
                return False
        
        # Prepare upload tasks: map created frames to local frame paths
        upload_tasks = []
        for idx, created_frame in enumerate(created_frames):
            if idx >= len(sorted_suffixes):
                break
            
            suffix = sorted_suffixes[idx]
            frame_data = frames_by_suffix[suffix]
            
            # Get upload URLs from created_frame
            high_url = created_frame.get('high_image', {}).get('url')
            low_url = created_frame.get('image', {}).get('url')
            blur_high_url = created_frame.get('blur_high_image', {}).get('url') if blur_people else None
            blur_low_url = created_frame.get('blur_image', {}).get('url') if blur_people else None
            
            # Add upload tasks
            if 'high' in frame_data and high_url:
                upload_tasks.append((high_url, frame_data['high']))
            if 'low' in frame_data and low_url:
                upload_tasks.append((low_url, frame_data['low']))
            if 'blurred_high' in frame_data and blur_high_url:
                upload_tasks.append((blur_high_url, frame_data['blurred_high']))
            if 'blurred_low' in frame_data and blur_low_url:
                upload_tasks.append((blur_low_url, frame_data['blurred_low']))
        
        # Upload images in parallel
        completed = 0
        failed_uploads = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(upload_image_binary, url, path): (url, path)
                for url, path in upload_tasks
            }
            
            for future in as_completed(future_to_task):
                url, path = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed_uploads.append((url, path))
                    
                    if completed % 10 == 0 or completed == len(upload_tasks):
                        update_status_callback(f"Uploading images: {completed}/{len(upload_tasks)}")
                except Exception as e:
                    logger.error(f"Error uploading image {path}: {e}")
                    failed_uploads.append((url, path))
        
        if failed_uploads:
            logger.warning(f"Failed to upload {len(failed_uploads)} images")
        
        logger.info(f"Uploaded {completed}/{len(upload_tasks)} image binaries")
        
        # Return frame objects (for compatibility)
        frame_objects = []
        for idx, created_frame in enumerate(created_frames):
            if idx >= len(sorted_suffixes):
                break
            
            frame_uuid = created_frame.get('frame', {}).get('uuid')
            if frame_uuid:
                frame_objects.append({
                    'uuid': frame_uuid,
                    'type': 'frame',
                    'suffix': sorted_suffixes[idx]
                })
        
        update_status_callback(f"Saved {len(frame_objects)} frames to cloud")
        logger.info(f"Created {len(frame_objects)} video frame records using batch-create")
        
        return frame_objects
    
    def store_video_frames(self, frame_collections: Dict[int, Dict[str, Optional[str]]], project_id: str, video_id: str, api_client, update_status_callback, layer_id: Optional[str] = None) -> List[str]:
        """Create video frame objects in API using batch-create endpoint.
        
        Note: Images are already uploaded, so we use batch-create to create frames and images,
        and return created frame UUIDs (no register/batch-update step).
        """
        saved_frame_ids: List[str] = []
        if not frame_collections:
            return saved_frame_ids
        
        if not layer_id:
            logger.error("Cannot create video frames: missing layer_id")
            return saved_frame_ids
        
        sorted_suffixes = sorted(frame_collections.keys())
        update_status_callback("Creating video frame objects...")
        
        # Calculate times_in_video for batch-create
        times_in_video = []
        suffix_to_images = {}
        for suffix in sorted_suffixes:
            images = frame_collections[suffix]
            high_id = images.get('high') or images.get('blurred_high')
            low_id = images.get('low') or images.get('blurred_low')
            
            if not high_id or not low_id:
                logger.warning(f"Skipping suffix {suffix}: missing high/low images")
                continue
            
            # Calculate time_in_video based on frame index and candidates_per_second FPS
            # Framet on poimittu candidates_per_second fps:llä (12 fps) ja valittu 1 fps:llä
            # Joten time_in_video = suffix / candidates_per_second
            time_in_video = suffix / float(self.candidates_per_second)
            times_in_video.append(time_in_video)
            suffix_to_images[suffix] = images
        
        if not times_in_video:
            update_status_callback("No frames to create")
            return saved_frame_ids
        
        # Check if blur images exist
        blur_people = any(
            images.get('blurred_high') or images.get('blurred_low')
            for images in frame_collections.values()
        )
        
        # Step 1: Create frames and images using batch-create
        update_status_callback(f"Creating {len(times_in_video)} video frames with batch-create...")
        created_frames = api_client.batch_create_video_frames(
            video_id=video_id,
            times_in_video=times_in_video,
            layer_id=layer_id,
            blur_people=blur_people
        )
        
        if not created_frames:
            logger.error("Batch-create video frames failed")
            return saved_frame_ids
        
        logger.info(f"Created {len(created_frames)} video frames with batch-create")

        for idx, created_frame in enumerate(created_frames):
            frame_uuid = created_frame.get('frame', {}).get('uuid')
            if not frame_uuid:
                logger.warning(f"Created frame missing UUID at index {idx}")
                continue
            saved_frame_ids.append(frame_uuid)

        update_status_callback(f"Created {len(saved_frame_ids)} video frames")
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
        Create frames for Stella VSLAM route calculation.
        These are separate from the selected frames used for API storage.
        
        Args:
            stitched_path: Path to stitched video
            update_status_callback: Status update callback
            
        Returns:
            List of frame paths for Stella
        """
        update_status_callback(f"Creating {self.candidates_per_second}fps frames for Stella route calculation...")
        
        stella_dir = self.work_dir / "stella_frames"
        stella_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames at low resolution (3840x1920) for Stella
        # Use candidates_per_second for frame rate
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
        """Use low-res frames for route calculation."""
        logger.info(f"Using {len(low_frames)} low res frames for route calculation...")
        update_status_callback(f"Using {len(low_frames)} low res frames for route calculation...")
        return low_frames
    
    def select_frames_from_extracted(self, all_high_frames: List[Path], all_low_frames: List[Path], update_status_callback) -> Tuple[List[Path], List[Path]]:
        """Select best frames (1 fps) from already extracted frames (candidates_per_second fps).
        
        Args:
            all_high_frames: All high-res frames extracted at candidates_per_second fps
            all_low_frames: All low-res frames extracted at candidates_per_second fps
            update_status_callback: Status update callback
        
        Returns:
            Tuple of (selected_high, selected_low) - best frames for API storage (1 fps)
        """
        update_status_callback("Selecting sharpest frames from each second...")
        
        # Select best frames based on sharpness (one per second)
        selected_suffixes = self._select_best_frames_by_sharpness(all_high_frames, self.candidates_per_second)
        
        # Filter frames to only selected ones
        def get_frame_suffix(frame_path: Path) -> Optional[int]:
            """Extract frame suffix number from filename."""
            try:
                stem = frame_path.stem
                if '_' in stem:
                    return int(stem.split("_")[-1])
                else:
                    # Try to extract number from end
                    match = re.search(r'(\d+)$', stem)
                    if match:
                        return int(match.group(1))
                return None
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not extract frame suffix from {frame_path.name}: {e}")
                return None
        
        selected_high = []
        selected_low = []
        
        for frame in all_high_frames:
            suffix = get_frame_suffix(frame)
            if suffix is not None and suffix in selected_suffixes:
                selected_high.append(frame)
        
        for frame in all_low_frames:
            suffix = get_frame_suffix(frame)
            if suffix is not None and suffix in selected_suffixes:
                selected_low.append(frame)
        
        logger.info(f"Filtered to {len(selected_high)} high and {len(selected_low)} low frames from {len(selected_suffixes)} selected suffixes")
        if len(selected_high) != len(selected_suffixes) or len(selected_low) != len(selected_suffixes):
            logger.warning(f"Mismatch: selected_suffixes={len(selected_suffixes)}, selected_high={len(selected_high)}, selected_low={len(selected_low)}")
            # Log some examples of selected_suffixes and what we found
            sample_suffixes = sorted(list(selected_suffixes))[:10]
            logger.warning(f"Sample selected_suffixes (first 10): {sample_suffixes}")
            # Check what frame suffixes we actually have
            high_suffixes = sorted([get_frame_suffix(f) for f in all_high_frames if get_frame_suffix(f) is not None])[:20]
            low_suffixes = sorted([get_frame_suffix(f) for f in all_low_frames if get_frame_suffix(f) is not None])[:20]
            logger.warning(f"Sample high frame suffixes (first 20): {high_suffixes}")
            logger.warning(f"Sample low frame suffixes (first 20): {low_suffixes}")
            # Check if selected_suffixes are in the frame lists
            missing_high = [s for s in sample_suffixes if s not in high_suffixes]
            missing_low = [s for s in sample_suffixes if s not in low_suffixes]
            if missing_high:
                logger.warning(f"Selected suffixes not found in high frames: {missing_high}")
            if missing_low:
                logger.warning(f"Selected suffixes not found in low frames: {missing_low}")
        
        # Move selected frames to selected directory
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
        
        update_status_callback(f"Selected {len(final_high)} best frames (1 fps)")
        return final_high, final_low
    
    def prepare_stella_frames_from_extracted(self, all_low_frames: List[Path], target_fps: float = 10.0, source_fps: float = 12.0) -> List[Path]:
        """Prepare frames for Stella from already extracted frames.
        
        Filters frames from source_fps to target_fps (e.g., 12 fps -> 10 fps).
        Moves frames to stella_frames directory with sequential naming for Stella.
        
        Args:
            all_low_frames: All low-res frames extracted at source_fps
            target_fps: Target FPS for Stella (default: 10.0)
            source_fps: Source FPS of extracted frames (default: 12.0)
        
        Returns:
            List of frame paths for Stella route calculation
        """
        if not all_low_frames:
            return []
        
        # Sort frames by frame number
        def get_frame_number(frame_path: Path) -> int:
            """Extract frame number from filename."""
            try:
                # Try to extract number from filename (e.g., "low_000123.jpg" -> 123)
                stem = frame_path.stem
                if '_' in stem:
                    return int(stem.split("_")[-1])
                else:
                    # If no underscore, try to extract number from end
                    match = re.search(r'(\d+)$', stem)
                    if match:
                        return int(match.group(1))
                    else:
                        logger.warning(f"Could not extract frame number from {frame_path.name}, using 0")
                        return 0
            except (ValueError, IndexError) as e:
                logger.warning(f"Error extracting frame number from {frame_path.name}: {e}, using 0")
                return 0
        
        # Filter out non-existent frames and sort
        existing_frames = [f for f in all_low_frames if f.exists()]
        if len(existing_frames) != len(all_low_frames):
            missing = len(all_low_frames) - len(existing_frames)
            logger.warning(f"{missing} frames are missing, using {len(existing_frames)} existing frames")
        
        if not existing_frames:
            logger.error("No existing frames found for Stella preparation")
            return []
        
        sorted_frames = sorted(existing_frames, key=get_frame_number)
        
        # Filter frames to target FPS
        filtered_frames = []
        frames_per_group = int(round(source_fps))
        frames_to_take = int(round(target_fps))
        
        # If target_fps >= source_fps, use all frames (no filtering needed)
        if frames_to_take >= frames_per_group:
            filtered_frames = sorted_frames
            logger.info(f"Using all {len(filtered_frames)} frames (target_fps {target_fps} >= source_fps {source_fps})")
        else:
            # Filter frames: take frames_to_take frames out of every frames_per_group
            # For example, 12 fps -> 10 fps: take 10 frames out of every 12
            for group_start in range(0, len(sorted_frames), frames_per_group):
                group = sorted_frames[group_start:group_start + frames_per_group]
                # Take first frames_to_take frames from each group
                filtered_frames.extend(group[:frames_to_take])
        
        # Copy frames to Stella directory with sequential naming
        stella_dir = self.work_dir / "stella_frames"
        stella_dir.mkdir(parents=True, exist_ok=True)
        
        stella_frames = []
        for idx, frame in enumerate(filtered_frames):
            try:
                if not frame.exists():
                    logger.warning(f"Frame {frame} does not exist, skipping")
                    continue
                
                stella_frame_path = stella_dir / f"stella_{idx:06d}.jpg"
                shutil.copy2(str(frame), str(stella_frame_path))
                stella_frames.append(stella_frame_path)
            except Exception as e:
                logger.error(f"Failed to copy frame {frame} to {stella_frame_path}: {e}")
                continue
        
        logger.info(f"Prepared {len(stella_frames)} frames for Stella (from {len(all_low_frames)} frames at {source_fps} fps -> {target_fps} fps)")
        return stella_frames

