"""
Frame processing module for extracting, selecting, and processing video frames.
"""

import logging
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class FrameProcessing:
    """Handles frame extraction, selection, and processing."""
    
    def __init__(self, work_dir: Path, candidates_per_second: int, blur_settings: Optional[Dict] = None):
        """Initialize frame processing."""
        self.work_dir = work_dir
        self.candidates_per_second = candidates_per_second
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
    
    def create_and_select_frames(self, stitched_path: Path, update_status_callback) -> Tuple[List[Path], List[Path]]:
        """Extract 12fps frames (high and low res), select sharpest from each group."""
        update_status_callback(f"Luodaan framet ({self.candidates_per_second}/s) stitchatusta tiedostosta...")
        
        high_dir = self.work_dir / "high_frames"
        low_dir = self.work_dir / "low_frames"
        
        # Extract high and low res frames
        high_frames, low_frames = self.extract_frames_high_and_low(
            stitched_path, high_dir, low_dir, self.candidates_per_second
        )
        
        if not high_frames or not low_frames:
            update_status_callback("Virhe: Framejen luonti epäonnistui")
            return [], []
        
        update_status_callback(f"Luotu {len(high_frames)} high res ja {len(low_frames)} low res framea")
        
        # Select best frames based on sharpness
        update_status_callback("Valitaan terävimmät framet jokaisesta sekunnista...")
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
        
        update_status_callback(f"Valittu {len(final_high)} parasta framea")
        return final_high, final_low
    
    def blur_frames_optional(self, high_frames: List[Path], low_frames: List[Path], blur_people: bool, update_status_callback) -> Tuple[List[Path], List[Path]]:
        """Optionally blur frames if blur_people is enabled."""
        if not blur_people:
            logger.info("Blur not enabled, skipping blur step")
            return high_frames, low_frames
        
        logger.info("Applying GPU-assisted blur to frames...")
        update_status_callback("Tehdään henkilötunnistus ja blurraus...")
        
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
            update_status_callback(f"Blurrattu {len(blurred_high)} framea")
            return blurred_high, blurred_low
        except Exception as e:
            logger.error(f"Frame blurring failed: {e}")
            update_status_callback(f"Virhe frame blurrauksessa: {str(e)}")
            return high_frames, low_frames
    
    def upload_frames_to_cloud(self, frame_paths: List[Path], project_id: str, video_id: Optional[str], api_client, update_status_callback) -> List[Dict]:
        """Save frames to cloud and create frame objects."""
        logger.info("Uploading frames to cloud...")
        update_status_callback(f"Tallennetaan {len(frame_paths)} framea pilveen ja luodaan frame objektit...")

        if not video_id:
            logger.error("Cannot upload frames: missing target video_id")
            update_status_callback("Virhe: video_id puuttuu, frameja ei voida tallentaa")
            return []
        
        frame_objects = []
        total = len(frame_paths)
        frame_collections: Dict[int, Dict[str, Optional[str]]] = defaultdict(lambda: {
            'high': None,
            'low': None,
            'blurred_high': None,
            'blurred_low': None
        })
        
        for i, frame_path in enumerate(frame_paths):
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
                        'path': str(frame_path)
                    }
                    frame_objects.append(frame_obj)
                    
                    # Update progress
                    if (i + 1) % 10 == 0 or i == total - 1:
                        update_status_callback(f"Tallennetaan frameja pilveen: {i + 1}/{total}")
                else:
                    logger.warning(f"Failed to store frame {i}")
                    
                suffix = self._extract_suffix(frame_path)
                if suffix is not None and image_id:
                    frame_collections[suffix][image_type] = image_id
                    
            except Exception as e:
                logger.error(f"Failed to upload frame {i}: {e}")
        
        logger.info(f"Uploaded {len(frame_objects)} frames")
        update_status_callback(f"Tallennettu {len(frame_objects)} framea pilveen")
        
        if video_id:
            video_frame_ids = self.store_video_frames(
                frame_collections,
                project_id,
                video_id,
                api_client,
                update_status_callback
            )
            logger.info(f"Created {len(video_frame_ids)} video frame records")
        
        return frame_objects
    
    def store_video_frames(self, frame_collections: Dict[int, Dict[str, Optional[str]]], project_id: str, video_id: str, api_client, update_status_callback) -> List[str]:
        """Create video frame objects in API using uploaded image references."""
        saved_frame_ids: List[str] = []
        if not frame_collections:
            return saved_frame_ids
        
        sorted_suffixes = sorted(frame_collections.keys())
        update_status_callback("Tallennetaan video frame objektit...")
        
        total = len(sorted_suffixes)
        for idx, suffix in enumerate(sorted_suffixes):
            images = frame_collections[suffix]
            high_id = images.get('high') or images.get('blurred_high')
            low_id = images.get('low') or images.get('blurred_low')
            blur_high_id = images.get('blurred_high')
            blur_low_id = images.get('blurred_low')
            
            if not high_id or not low_id:
                logger.debug(f"Skipping suffix {suffix}: missing high/low images")
                continue
            
            time_in_video = suffix / float(self.candidates_per_second)
            frame_uuid = api_client.save_videoframe(
                project_id=project_id,
                video_id=video_id,
                time_in_video=time_in_video,
                high_res_image_id=high_id,
                low_res_image_id=low_id,
                blur_high_image_id=blur_high_id,
                blur_low_image_id=blur_low_id
            )
            
            if frame_uuid:
                saved_frame_ids.append(frame_uuid)
                if (idx + 1) % 5 == 0 or idx == total - 1:
                    update_status_callback(f"Tallennetaan video frameja: {idx + 1}/{total}")
            else:
                logger.warning(f"Failed to create video frame for suffix {suffix}")
        
        update_status_callback(f"Tallennettu {len(saved_frame_ids)} video framea API:iin")
        return saved_frame_ids
    
    def get_route_frames_from_low_res(self, low_frames: List[Path], update_status_callback) -> List[Path]:
        """Use 12fps low res frames for route calculation."""
        logger.info(f"Using {len(low_frames)} low res frames for route calculation...")
        update_status_callback(f"Käytetään {len(low_frames)} low res framea reitin laskentaan...")
        return low_frames

