"""
API client module for handling all API requests.
"""

import logging
import math
import time
import requests
import tempfile
from typing import Dict, Optional, List, Any, Tuple
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

try:
    import nudged
except ImportError:
    nudged = None
    logger.warning("nudged library not available. Install with: pip install nudged")

logger = logging.getLogger(__name__)


class MediaServerAPIClient:
    """Handles all Media Server API communication."""

    def __init__(self, api_domain: str, api_key: str, worker_id: str):
        """Initialize Media Server API client."""
        self.api_domain = api_domain
        self.api_key = api_key
        self.worker_id = worker_id
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _api_request(self, method: str, endpoint: str, max_retries: int = 3, **kwargs) -> Optional[Dict]:
        """
        Make API request with error handling and retry logic.
        
        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint path
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments passed to requests.request()
        
        Returns:
            Response JSON as dict if successful, None otherwise
        """
        url = f"{self.api_domain}{endpoint}"
        
        for attempt in range(max_retries):
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
                is_last_attempt = (attempt == max_retries - 1)
                
                # Log error details
                error_msg = f"Media Server API request failed (attempt {attempt + 1}/{max_retries}): {e}"

                # Log response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        response_text = e.response.text
                        status_code = e.response.status_code
                        if is_last_attempt:
                            logger.error(f"Media Server API response status: {status_code}, text: {response_text[:500]}")
                        else:
                            logger.warning(f"Media Server API response status: {status_code}, text: {response_text[:200]}")
                    except Exception:
                        pass
                raise Exception(error_msg) from e

    def fetch_next_task(self, reset: bool = False) -> Optional[Dict]:
        """Fetch next video_task from Media Server API."""
        # Keep polling noise out of INFO logs; the caller (processing_runner) logs one INFO line per poll.
        logger.debug("Fetching next video task from Media Server API...")
        payload = {
            "task_type": "process_360_video", 
            "environment": "development,production",
            "worker_id": self.worker_id,
            "change_status": "false" if reset else "true"
        }
        endpoint = f"/api/tasks/fetch"
        task = self._api_request('POST', endpoint, json=payload)
        if task:
            # Task payload can be large; avoid logging full token/details at INFO.
            task_id = task.get('task_id') or task.get('uuid') or task.get('id')
            logger.debug(f"Fetched task from Media Server API (task_id={task_id})")
            return task
        else:
            logger.debug("No task found")
            return None
    
    def update_task_status(self, task_id: str, status: str, result: Optional[str] = None) -> bool:
        """Update task status in Media Server API."""
        endpoint = f"/api/tasks/{task_id}/"
        payload = {"status": status}
        if result is not None:
            payload["result"] = result
        self._api_request('PATCH', endpoint, json=payload)
        if result is not None:
            logger.info(f"Task {task_id} status set succesfully to {status} with result {result}")
        else:
            logger.info(f"Task {task_id} status set succesfully to {status} without result")
        return True


class APIClient:
    """Handles all AIC API communication."""
    
    def __init__(self, api_domain: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize API client."""
        self.api_domain = api_domain or ''
        self.api_key = api_key or ''
        self.headers = {
            'Authorization': f'Token {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json'
        }
        self.current_task_id = None
        self.current_video_recording_id = None
        self.test_mode = False
    
    def update_credentials(self, api_domain: str, api_key: str):
        """Update API domain and key, refreshing headers."""
        self.api_domain = api_domain
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        logger.info(f"Updated API client credentials (domain: {api_domain})")
    
    def _api_request(self, method: str, endpoint: str, max_retries: int = 3, **kwargs) -> Optional[Dict]:
        """
        Make API request with error handling and retry logic.
        
        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint path
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments passed to requests.request()
        
        Returns:
            Response JSON as dict if successful, None otherwise
        """
        url = f"{self.api_domain}{endpoint}"
        
        # Prepare headers - if using 'data' parameter (form-data), don't set Content-Type
        # requests will set it automatically to application/x-www-form-urlencoded
        headers = self.headers.copy()
        if 'data' in kwargs and 'json' not in kwargs:
            # Remove Content-Type header to let requests set it automatically for form-data
            headers.pop('Content-Type', None)
        
        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs
                )
                response.raise_for_status()
                return response.json() if response.content else {}
                
            except requests.exceptions.RequestException as e:
                is_last_attempt = (attempt == max_retries - 1)
                
                # Get status code if available
                status_code = None
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                
                # Log error details
                error_msg = f"API request failed (attempt {attempt + 1}/{max_retries}): {e}"
                if is_last_attempt:
                    logger.error(error_msg)
                else:
                    logger.warning(error_msg)
                
                # Log response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        response_text = e.response.text
                        if is_last_attempt:
                            logger.error(f"API response status: {status_code}, text: {response_text[:500]}")
                        else:
                            logger.warning(f"API response status: {status_code}, text: {response_text[:200]}")
                    except Exception:
                        pass
                
                # Don't retry on last attempt
                if is_last_attempt:
                    return None
                
                # Use longer backoff for server errors (5xx) like 502 Bad Gateway
                if status_code and status_code >= 500:
                    # Longer delay for server errors: 3s, 6s, 12s
                    delay = 3 * (2 ** attempt)
                else:
                    # Standard exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Should not reach here, but just in case
        return None
    
    def fetch_video_recording(self, video_recording_id: str) -> Optional[Dict]:
        """Fetch video-recording data from API."""
        logger.info(f"Fetching video-recording data for {video_recording_id}...")
        return self._api_request('GET', f'/api/v1/video-recording/{video_recording_id}/')
    
    def fetch_layer(self, layer_id: str) -> Optional[Dict]:
        """Fetch layer data from API including image dimensions and corner GPS coordinates."""
        logger.info(f"Fetching layer data for {layer_id}...")
        return self._api_request('GET', f'/api/v1/layer/{layer_id}/')
    
    def store_image(self, project_id: str, image_type: str, image_size: int, image_binary: bytes) -> Optional[str]:
        """
        Store image binary to cloud storage.
        
        Process:
        1. POST /api/v1/image/ {project, size, type} -> {uuid, upload_url, ...}
        2. PUT upload_url with image_binary -> code 200
        3. PUT /api/v1/image/<image_uuid>/register_file
        
        Returns image_id (uuid) if successful, None otherwise.
        """
        try:
            # Step 1: Get upload URL from API
            logger.info(f"Requesting upload URL for image (type: {image_type}, size: {image_size})...")
            upload_response = self._api_request(
                'POST',
                '/api/v1/image/',
                json={
                    'project': project_id,
                    'name': 'image',
                    'size': image_size,
                    'type': image_type
                }
            )
            
            if not upload_response:
                logger.error("Failed to get upload URL from API")
                return None
            
            image_uuid = upload_response.get('uuid')
            upload_url = upload_response.get('upload_url')
            
            if not image_uuid or not upload_url:
                logger.error("Missing uuid or upload_url in API response")
                return None
            
            logger.info(f"Got upload URL for image {image_uuid}")
            
            # Step 2: Upload image binary to upload_url
            logger.info(f"Uploading image binary to storage...")
            try:
                upload_result = requests.put(
                    upload_url,
                    data=image_binary,
                    headers={'Content-Type': 'application/octet-stream'},
                    timeout=300
                )
                upload_result.raise_for_status()
                
                if upload_result.status_code != 200:
                    logger.error(f"Upload failed with status code {upload_result.status_code}")
                    logger.error(f"Response text: {upload_result.text}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Image upload failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        logger.error(f"Response text: {e.response.text}")
                    except Exception:
                        pass
                return None
            
            logger.info(f"Image binary uploaded successfully")
            
            # Step 3: Register file with API
            logger.info(f"Registering file for image {image_uuid}...")
            register_response = self._api_request(
                'PUT',
                f'/api/v1/image/{image_uuid}/register_file'
            )
            
            if not register_response:
                logger.error("Failed to register file with API")
                return None
            
            logger.info(f"Image stored successfully with ID: {image_uuid}")
            
            return image_uuid
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to store image: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error storing image: {e}")
            return None

    def store_zip_file(
        self,
        project_id: str,
        zip_binary: bytes,
        name: str = 'stella_frames.zip',
        category: Optional[str] = None,
        video_recording_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Store zip file to cloud storage.
        
        Process:
        1. POST /api/v1/file/ {project, size, name, category?} -> {uuid, upload_url, ...}
        2. PUT upload_url with zip_binary -> code 200
        3. PUT /api/v1/file/<file_uuid>/register_file
        """
        try:
            file_size = len(zip_binary)
            logger.info(f"Requesting upload URL for zip file (size: {file_size})...")
            payload = {
                'project': project_id,
                'name': name,
                'size': file_size
            }
            if category:
                payload['category'] = category
            if video_recording_id:
                payload['recording'] = video_recording_id
            
            upload_response = self._api_request(
                'POST',
                '/api/v1/file/',
                json=payload
            )
            
            if not upload_response:
                logger.error("Failed to get upload URL for zip file")
                return None
            
            file_uuid = upload_response.get('uuid')
            upload_url = upload_response.get('upload_url')
            
            if not file_uuid or not upload_url:
                logger.error("Missing uuid or upload_url in zip file response")
                return None
            
            logger.info(f"Got upload URL for zip file {file_uuid}")
            
            try:
                upload_result = requests.put(
                    upload_url,
                    data=zip_binary,
                    headers={'Content-Type': 'application/zip'},
                    timeout=600
                )
                upload_result.raise_for_status()
                
                if upload_result.status_code not in (200, 201):
                    logger.error(f"Zip upload failed with status code {upload_result.status_code}")
                    logger.error(f"Response text: {upload_result.text}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Zip upload failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        logger.error(f"Response text: {e.response.text}")
                    except Exception:
                        pass
                return None
            
            logger.info("Zip file uploaded successfully")
            
            register_response = self._api_request(
                'PUT',
                f'/api/v1/file/{file_uuid}/register_file'
            )
            
            if not register_response:
                logger.error("Failed to register zip file with API")
                return None
            
            logger.info(f"Zip file stored successfully with ID: {file_uuid}")
            return file_uuid
        
        except Exception as e:
            logger.error(f"Failed to store zip file: {e}")
            return None

    def store_video(
        self,
        project_id: str,
        video_type: str,
        video_size: int,
        video_binary: bytes,
        video_recording_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Optional[str]:
        """
        Store video binary to cloud storage.
        
        Process:
        1. POST /api/v1/video/ {project, size, type} -> {uuid, upload_url, ...}
        2. PUT upload_url with video_binary -> code 200
        3. PUT /api/v1/video/<video_uuid>/register_file
        
        Returns video_id (uuid) if successful, None otherwise.
        """
        try:
            # Step 1: Get upload URL from API
            logger.info(f"Requesting upload URL for video (type: {video_type}, size: {video_size})...")
            upload_response = self._api_request(
                'POST',
                '/api/v1/video/',
                json={
                    'project': project_id,
                    'name': name or 'video',
                    'size': video_size,
                    'category': 'video_360_field_raw',
                    'type': video_type,
                    **({'recording': video_recording_id} if video_recording_id else {})
                }
            )
            
            if not upload_response:
                logger.error("Failed to get upload URL from API")
                return None
            
            video_uuid = upload_response.get('uuid')
            upload_url = upload_response.get('upload_url')
            
            if not video_uuid or not upload_url:
                logger.error("Missing uuid or upload_url in API response")
                return None
            
            logger.info(f"Got upload URL for video {video_uuid}")
            
            # Step 2: Upload video binary to upload_url (skip if binary is empty)
            if video_size > 0 and len(video_binary) > 0:
                logger.info(f"Uploading video binary to storage...")
                try:
                    upload_result = requests.put(
                        upload_url,
                        data=video_binary,
                        headers={'Content-Type': 'application/octet-stream'},
                        timeout=600  # Longer timeout for videos
                    )
                    upload_result.raise_for_status()
                    
                    if upload_result.status_code != 200:
                        logger.error(f"Upload failed with status code {upload_result.status_code}")
                        logger.error(f"Response text: {upload_result.text}")
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error(f"Video upload failed: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            logger.error(f"Response text: {e.response.text}")
                        except Exception:
                            pass
                    return None
                
                logger.info(f"Video binary uploaded successfully")
                
                # Step 3: Register file with API (only if we uploaded binary)
                logger.info(f"Registering file for video {video_uuid}...")
                register_response = self._api_request(
                    'PUT',
                    f'/api/v1/video/{video_uuid}/register_file',
                )
                
                if not register_response:
                    logger.error("Failed to register file with API")
                    return None
            else:
                logger.info(f"Skipping video binary upload (size={video_size}, binary_length={len(video_binary)})")
            
            logger.info(f"Video stored successfully with ID: {video_uuid}")
            return video_uuid
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to store video: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error storing video: {e}")
            return None
    
    def save_videoframe(
        self,
        project_id: str,
        video_id: str,
        time_in_video: float,
        high_res_image_id: str,
        low_res_image_id: str,
        blur_high_image_id: Optional[str] = None,
        blur_low_image_id: Optional[str] = None,
        layer_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Save video frame object to API.
        
        Args:
            project_id: Project UUID
            video_id: Video UUID
            time_in_video: Time in video in seconds
            high_res_image_id: High resolution image UUID
            low_res_image_id: Low resolution image UUID
            blur_high_image_id: Optional blurred high image UUID
            blur_low_image_id: Optional blurred low image UUID
            layer_id: Optional layer UUID
        
        Returns:
            frame_uuid if successful, None otherwise.
        """
        try:
            logger.info(f"Saving video frame (video: {video_id}, time: {time_in_video}s)...")
            # API expects 'high_image' and 'image' (not 'high_res_image' and 'low_res_image')
            payload = {
                'project': project_id,
                'video': video_id,
                'time_in_video': math.floor(time_in_video),
                'high_image': high_res_image_id,
                'image': low_res_image_id
            }
            
            if blur_high_image_id:
                payload['blur_high_image'] = blur_high_image_id
            if blur_low_image_id:
                payload['blur_image'] = blur_low_image_id
            
            if layer_id:
                payload['camera_layer_position'] = {
                    "rx": 0,
                    "ry": 0,
                    "rz": 0,
                    "layer": layer_id,
                    #"yaw": 0,
                    #"roll": 0,
                    #"pitch": 0,
                }
            
            response = self._api_request(
                'POST',
                '/api/v1/video-frame/',
                json=payload
            )
            
            if not response:
                logger.error("Failed to save video frame")
                return None
            
            frame_uuid = response.get('uuid')
            if not frame_uuid:
                logger.error("Missing uuid in API response")
                return None
            
            # In test mode, print the full API response
            if self.test_mode:
                import json as json_module
                print(f"\n=== VIDEO-FRAME API RESPONSE (time: {time_in_video}s) ===")
                print(json_module.dumps(response, indent=2))
                print("=" * 60)
                
                # Check for missing image fields
                missing_images = []
                # We send 'high_image' and API should return it
                if not response.get('high_image'):
                    missing_images.append('high_image')
                # We send 'image' and API should return it
                if not response.get('image'):
                    missing_images.append('image')
                if blur_high_image_id and not response.get('blur_high_image'):
                    missing_images.append('blur_high_image')
                if blur_low_image_id and not response.get('blur_image'):
                    missing_images.append('blur_image')
                
                if missing_images:
                    print(f"⚠️  WARNING: Missing image fields: {', '.join(missing_images)}")
                    print(f"   Sent: high_image={high_res_image_id}, image={low_res_image_id}")
                    print(f"   Sent: blur_high_image={blur_high_image_id}, blur_image={blur_low_image_id}")
                    print(f"   Payload sent: {json_module.dumps(payload, indent=2)}")
                    print("=" * 60)
            
            logger.info(f"Video frame saved successfully with ID: {frame_uuid}")
            return frame_uuid
            
        except Exception as e:
            logger.error(f"Failed to save video frame: {e}")
            return None

    def save_video_frames_bulk(self, frames_payload: List[Dict]) -> Optional[List[Dict]]:
        """Save multiple video frames using bulk endpoint."""
        if not frames_payload:
            return []
        
        logger.info(f"Saving {len(frames_payload)} video frames in bulk...")
        response = self._api_request(
            'POST',
            '/api/v1/video-frame/bulk/',
            json=frames_payload
        )
        
        if response is None:
            logger.error("Bulk video frame request failed")
            return None
        
        if isinstance(response, list):
            logger.info("Bulk video frame request completed")
            return response
        
        logger.warning(f"Unexpected response format from bulk video frame endpoint: {type(response)}")
        return None
    
    def batch_create_video_frames(self, video_id: str, times_in_video: List[float], layer_id: str, blur_people: bool = False) -> Optional[List[Dict]]:
        """Create video frames and images using batch-create endpoint.
        
        Args:
            video_id: Video UUID
            times_in_video: List of times in video (seconds)
            layer_id: Layer UUID
            blur_people: Whether to create blur image slots
        
        Returns:
            List of frame dicts with frame uuid, image uuid+url, high_image uuid+url, etc.
        """
        if not times_in_video:
            return []
        
        logger.info(f"Creating {len(times_in_video)} video frames using batch-create endpoint...")
        
        # Convert times to comma-separated string
        times_str = ",".join([str(int(t)) for t in times_in_video])
        
        # Prepare form data
        form_data = {
            'video': video_id,
            'times_in_video': times_str,
            'layer': layer_id,
            'blur_people': 'true' if blur_people else 'false'
        }
        
        response = self._api_request(
            'POST',
            '/api/v1/video-frame/batch-create/',
            data=form_data
        )
        
        if response is None:
            logger.error("Batch-create video frames request failed")
            return None
        
        frames = response.get('frames', [])
        if frames:
            logger.info(f"Created {len(frames)} video frames with batch-create")
        return frames
    
    def batch_update_video_frames(self, frames_data: List[Dict]) -> bool:
        """Update video frames using batch-update endpoint.
        
        Args:
            frames_data: List of frame update dicts, each containing:
                - frame: frame uuid
                - image/high_image/blur_image/blur_high_image: {uuid, register: true/false}
                - camera_layer_position: {rx, ry}
                - hotspots: (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not frames_data:
            return True
        
        logger.info(f"Updating {len(frames_data)} video frames using batch-update endpoint...")
        
        # Prepare form data with JSON-encoded frames_data
        import json as json_module
        form_data = {
            'frames_data': json_module.dumps(frames_data)
        }
        
        response = self._api_request(
            'POST',
            '/api/v1/video-frame/batch-update/',
            data=form_data
        )
        
        if response is None:
            logger.error("Batch-update video frames request failed")
            return False
        
        status = response.get('status')
        if status == 'updated':
            logger.info(f"Updated {len(frames_data)} video frames with batch-update")
            return True
        
        error = response.get('error', 'Unknown error')
        logger.error(f"Batch-update failed: {error}")
        return False

    def fetch_video_frames(self, video_id: str, limit: int = 200) -> List[Dict]:
        """Fetch all video-frame records for a given video."""
        frames: List[Dict] = []
        offset = 0
        while True:
            params = {'video': video_id, 'offset': offset, 'limit': limit}
            logger.info(f"Fetching video frames with params: {params}")
            response = self._api_request('GET', '/api/v1/video-frame/', params=params)
            if not response:
                break
            if isinstance(response, dict) and 'results' in response:
                results = response.get('results') or []
                frames.extend(results)
                if response.get('next'):
                    offset += limit
                    logger.info(f"Fetching next page of video frames with offset: {offset}")
                    time.sleep(0.5)
                    continue
                break
            if isinstance(response, list):
                frames.extend(response)
                break
            # Unexpected shape, stop to avoid infinite loop
            break
        logger.info(f"Fetched {len(frames)} video frames for video {video_id}")
        return frames

    def delete_image(self, image_id: Optional[str]) -> bool:
        """Delete image asset from API."""
        if not image_id:
            return False
        result = self._api_request('DELETE', f'/api/v1/image/{image_id}/')
        if result is None:
            logger.warning(f"Failed to delete image {image_id}")
            return False
        logger.info(f"Deleted image {image_id}")
        return True

    def delete_video_frame(self, frame_id: Optional[str]) -> bool:
        """Delete video-frame record."""
        if not frame_id:
            return False
        result = self._api_request('DELETE', f'/api/v1/video-frame/{frame_id}/')
        if result is None:
            logger.warning(f"Failed to delete video frame {frame_id}")
            return False
        logger.info(f"Deleted video frame {frame_id}")
        return True
    
    def update_video_frame_camera_position(self, frame_id: str, rx: float, ry: float, layer_id: str) -> bool:
        """Update video frame camera_layer_position with rx, ry, and layer.
        
        Args:
            frame_id: Video frame UUID
            rx: X coordinate (rotation/position)
            ry: Y coordinate (rotation/position)
            layer_id: Layer UUID
        
        Returns:
            True if successful, False otherwise
        """
        if not frame_id:
            logger.warning("Cannot update video frame: missing frame_id")
            return False
        
        payload = {
            'camera_layer_position': {
                'rx': rx,
                'ry': ry,
                'layer': layer_id
            }
        }
        
        result = self._api_request(
            'PATCH',
            f'/api/v1/video-frame/{frame_id}/',
            json=payload
        )
        
        if result is not None:
            logger.debug(f"Updated video frame {frame_id} camera position: rx={rx}, ry={ry}")
            return True
        else:
            logger.warning(f"Failed to update video frame {frame_id} camera position")
            return False
    
    def update_video_frames_from_raw_path(self, video_id: str, raw_path: Dict, layer_id: str, update_status_callback=None, start_position: Optional[Dict] = None, end_position: Optional[Dict] = None) -> int:
        """Update all video frames with camera_layer_position from raw_path using nudged transformation.
        
        Args:
            video_id: Video UUID
            raw_path: Dict with structure {"0": [0.0, "timestamp x y z ..."], "1": [1.0, "..."], ...}
            layer_id: Layer UUID
            update_status_callback: Optional status update callback
            start_position: Optional dict with 'rx' and 'ry' keys for start position mapping
            end_position: Optional dict with 'rx' and 'ry' keys for end position mapping
        
        Returns:
            Number of frames updated successfully
        """
        if not video_id or not raw_path:
            logger.warning("Cannot update video frames: missing video_id or raw_path")
            return 0
        
        if nudged is None:
            logger.error("nudged library not available. Cannot update video frames with advanced mapping.")
            return 0
        
        try:
            # Fetch layer data to get image dimensions and corner GPS coordinates
            layer_data = None
            if layer_id:
                try:
                    layer_data = self.fetch_layer(layer_id)
                    if not layer_data:
                        logger.warning(f"Could not fetch layer data for {layer_id}, falling back to simple normalization")
                    elif not isinstance(layer_data, dict):
                        logger.warning(f"Layer data is not a dict (type: {type(layer_data)}, value: {layer_data}), falling back to simple normalization")
                        layer_data = None
                except Exception as e:
                    logger.warning(f"Error fetching layer data: {e}, falling back to simple normalization")
                    layer_data = None
            
            # Extract x and z coordinates from raw_path (y is vertical in Stella)
            def get_points_coordinates_from_stella_results_line(stella_results_line: str) -> Tuple[float, float]:
                """Extract x and z coordinates from Stella results line."""
                split_line = stella_results_line.split()
                if len(split_line) < 4:
                    raise ValueError(f"Invalid raw_path line: {stella_results_line[:50]}")
                return float(split_line[1]), float(split_line[3])  # x and z coordinates
        
            # Convert raw_path dict to list of (x, z) tuples
            raw_path_list = list(raw_path.values())
            raw_path_xz = []
            for i, p in enumerate(raw_path_list):
                if not p or len(p) < 2:
                    continue
                try:
                    raw_xz = get_points_coordinates_from_stella_results_line(p[1])
                    raw_path_xz.append(raw_xz)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse raw_path entry {i}: {e}")
                    continue
            
            if not raw_path_xz:
                logger.warning("No valid coordinates found in raw_path")
                return 0
            
            # Determine starting and ending positions
            starting_position = None
            ending_position = None
            
            # Log what we received
            logger.info(f"Received start_position: {start_position} (type: {type(start_position)})")
            logger.info(f"Received end_position: {end_position} (type: {type(end_position)})")
            
            # Check if start_position and end_position are dicts and have required keys
            if (start_position and isinstance(start_position, dict) and 
                end_position and isinstance(end_position, dict) and
                'rx' in start_position and 'ry' in start_position and 
                'rx' in end_position and 'ry' in end_position):
                try:
                    starting_position = (float(start_position['rx']), float(start_position['ry']))
                    ending_position = (float(end_position['rx']), float(end_position['ry']))
                    logger.info(f"Parsed starting_position: {starting_position}, ending_position: {ending_position}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing start_position/end_position: {e}, using defaults")
                    starting_position = (0.5, 0.5)
                    ending_position = (0.5, 0.5)
            else:
                # Use default 0.5 for both if not specified
                starting_position = (0.5, 0.5)
                ending_position = (0.5, 0.5)
                if not start_position:
                    logger.warning("start_position is None or missing")
                elif not isinstance(start_position, dict):
                    logger.warning(f"start_position is not a dict, got {type(start_position)}: {start_position}")
                elif 'rx' not in start_position or 'ry' not in start_position:
                    logger.warning(f"start_position dict missing 'rx' or 'ry' keys, got keys: {start_position.keys()}")
                
                if not end_position:
                    logger.warning("end_position is None or missing")
                elif not isinstance(end_position, dict):
                    logger.warning(f"end_position is not a dict, got {type(end_position)}: {end_position}")
                elif 'rx' not in end_position or 'ry' not in end_position:
                    logger.warning(f"end_position dict missing 'rx' or 'ry' keys, got keys: {end_position.keys()}")
                
                logger.info("Using default positions (0.5, 0.5) for both start and end")
            
            # If start and end are the same, log warning but keep the values
            # Don't override user-provided positions even if they're the same
            if starting_position == ending_position:
                logger.warning("start_position and end_position are the same - this may cause issues with route mapping")
            
            # Define get_rx_ry function before using it in transformation verification
            def get_rx_ry(pixel_coordinate: List[float], img_width: Optional[int] = None, img_height: Optional[int] = None) -> Tuple[float, float]:
                """Return rx, ry values (may be outside 0.0-1.0 range to preserve route shape).
                
                Note: pixel_coordinate contains [x, z] from Stella, but we need to swap them:
                - rx = z (horizontal position on layer)
                - ry = x (vertical position on layer)
                """
                # Swap x and z coordinates: rx = z, ry = x
                rx, ry = pixel_coordinate[1], pixel_coordinate[0]
                # Don't clamp - allow values outside 0-1 range to preserve route shape
                # Server should accept any float values
                return float(rx), float(ry)
            
            # Try to use similarity transformation (nudged) to preserve route shape
            # If that fails or nudged is not available, use raw coordinates directly
            cartesian_path = []
            use_similarity_transform = False
            
            if nudged is not None and len(raw_path_xz) >= 2:
                try:
                    # Get first and last points from raw_path (Stella coordinates)
                    x1_stella = raw_path_xz[0][0]
                    z1_stella = raw_path_xz[0][1]
                    xN_stella = raw_path_xz[-1][0]
                    zN_stella = raw_path_xz[-1][1]
                    
                    # Calculate Stella distance
                    stella_dx = xN_stella - x1_stella
                    stella_dz = zN_stella - z1_stella
                    stella_distance = math.sqrt(stella_dx**2 + stella_dz**2)
                    
                    # Map coordinates: get_rx_ry swaps (rx = z, ry = x)
                    # So Stella [x, z] maps to map [ry, rx]
                    map_rx1 = starting_position[0]  # map start rx
                    map_ry1 = starting_position[1]  # map start ry
                    map_rxN = ending_position[0]    # map end rx
                    map_ryN = ending_position[1]    # map end ry
                    
                    # Calculate map distance
                    map_drx = map_rxN - map_rx1  # corresponds to stella z
                    map_dry = map_ryN - map_ry1  # corresponds to stella x
                    map_distance = math.sqrt(map_drx**2 + map_dry**2)
                    
                    # Calculate scale factor explicitly: map_distance / stella_distance
                    if stella_distance > 0:
                        scale_factor = map_distance / stella_distance
                        logger.info(f"Route transformation: Stella distance={stella_distance:.6f}, Map distance={map_distance:.6f}, Scale factor={scale_factor:.6f}")
                    else:
                        logger.warning("Stella start/end distance is zero, cannot calculate scale")
                        scale_factor = 1.0
                        use_similarity_transform = False
                        raise ValueError("Stella distance is zero")
                    
                    # Prepare points for similarity transformation
                    # domain_points: Stella coordinates [x, z]
                    # range_points: Map coordinates [ry, rx] (swapped because get_rx_ry swaps)
                    domain_points = [[x1_stella, z1_stella], [xN_stella, zN_stella]]
                    range_points = [[map_ry1, map_rx1], [map_ryN, map_rxN]]
                    
                    # Estimate similarity transformation (handles scale, rotation, translation)
                    trans = nudged.estimate(domain_points, range_points)
                    
                    # Verify the transformation scale matches our calculation
                    trans_scale = trans.get_scale() if hasattr(trans, 'get_scale') else None
                    if trans_scale:
                        scale_diff = abs(trans_scale - scale_factor)
                        logger.info(f"Transformation scale: {trans_scale:.6f} (expected: {scale_factor:.6f}, diff: {scale_diff:.6f})")
                        
                        # If scale differs significantly, apply scale correction
                        if scale_diff > 0.0001:
                            logger.warning(f"Scale mismatch detected! Transformation scale ({trans_scale:.6f}) differs from calculated scale ({scale_factor:.6f})")
                            # Apply scale correction: manually apply scale, rotation, translation
                            trans_rotation = trans.get_rotation() if hasattr(trans, 'get_rotation') else 0.0
                            
                            # Transform all points manually with corrected scale
                            cartesian_path = []
                            for point in raw_path_xz:
                                # 1. Translate to origin (relative to start point)
                                rel_x = point[0] - x1_stella
                                rel_z = point[1] - z1_stella
                                
                                # 2. Apply scale
                                scaled_x = rel_x * scale_factor
                                scaled_z = rel_z * scale_factor
                                
                                # 3. Apply rotation (from transformation)
                                cos_r = math.cos(trans_rotation)
                                sin_r = math.sin(trans_rotation)
                                rotated_x = scaled_x * cos_r - scaled_z * sin_r
                                rotated_z = scaled_x * sin_r + scaled_z * cos_r
                                
                                # 4. Translate to map start position (remember: rx = z, ry = x)
                                final_ry = rotated_x + map_ry1
                                final_rx = rotated_z + map_rx1
                                
                                # Store as [x, z] - get_rx_ry will swap them to [rx, ry]
                                cartesian_path.append([final_ry, final_rx])
                            
                            logger.info("Applied scale correction to route transformation")
                        else:
                            # Use transformation as-is if scale matches
                            for point in raw_path_xz:
                                transformed = trans.transform(point)
                                cartesian_path.append([float(transformed[0]), float(transformed[1])])
                    else:
                        # If we can't get scale, use transformation as-is
                        for point in raw_path_xz:
                            transformed = trans.transform(point)
                            cartesian_path.append([float(transformed[0]), float(transformed[1])])
                    
                    # Verify transformation: check start and end points
                    if cartesian_path:
                        first_transformed = cartesian_path[0]
                        first_rx_ry = get_rx_ry(first_transformed)
                        last_transformed = cartesian_path[-1]
                        last_rx_ry = get_rx_ry(last_transformed)
                        
                        logger.info(f"Transformation verification:")
                        logger.info(f"  Start: [{first_rx_ry[0]:.6f}, {first_rx_ry[1]:.6f}] (expected: [{map_rx1:.6f}, {map_ry1:.6f}])")
                        logger.info(f"  End:   [{last_rx_ry[0]:.6f}, {last_rx_ry[1]:.6f}] (expected: [{map_rxN:.6f}, {map_ryN:.6f}])")
                        
                        # Check accuracy
                        start_error = math.sqrt((first_rx_ry[0] - map_rx1)**2 + (first_rx_ry[1] - map_ry1)**2)
                        end_error = math.sqrt((last_rx_ry[0] - map_rxN)**2 + (last_rx_ry[1] - map_ryN)**2)
                        if start_error > 0.001 or end_error > 0.001:
                            logger.warning(f"Transformation accuracy: start error={start_error:.6f}, end error={end_error:.6f}")
                    
                    use_similarity_transform = True
                    logger.info("Route transformed: scaled, rotated, and aligned to map using start/end points")
                    
                except Exception as e:
                    logger.warning(f"Similarity transformation failed: {e}, using raw coordinates")
                    use_similarity_transform = False
            
            # Fallback: use raw coordinates directly without normalization
            # This preserves the exact route shape but values may be outside 0-1 range
            if not use_similarity_transform:
                logger.info("Using raw_path coordinates directly (no normalization) to preserve route shape")
                # Use raw coordinates as-is, just convert to float
                for point in raw_path_xz:
                    cartesian_path.append([float(point[0]), float(point[1])])
            
            # Fetch all video frames
            if update_status_callback:
                update_status_callback("Fetching video frames for camera position update...")
            frames = self.fetch_video_frames(video_id)
            
            if not frames:
                logger.warning(f"No video frames found for video {video_id}")
                return 0
            
            # Sort frames by time_in_video
            frames.sort(key=lambda f: f.get('time_in_video', 0))
            
            # Check if number of frames matches raw_path entries
            if len(frames) != len(raw_path_xz):
                logger.warning(f"Number of frames ({len(frames)}) does not match raw_path entries ({len(raw_path_xz)})")
                # Use time-based matching instead
                logger.info("Using time-based matching for frames")
            
            logger.info(f"Updating {len(frames)} video frames with camera positions from raw_path using batch-update...")
            
            # Prepare batch-update data
            update_data = []
            failed_count = 0
            
            for idx, frame in enumerate(frames):
                time_in_video = frame.get('time_in_video')
                if time_in_video is None:
                    logger.warning(f"Video frame {frame.get('uuid')} missing time_in_video, skipping")
                    failed_count += 1
                    continue
                
                # Get corresponding raw_path entry
                if len(frames) == len(raw_path_xz):
                    # Direct index matching
                    cartesian_point = cartesian_path[idx]
                else:
                    # Time-based matching
                    second_index = int(math.floor(time_in_video))
                    if second_index < len(cartesian_path):
                        cartesian_point = cartesian_path[second_index]
                    else:
                        logger.debug(f"No raw_path data for time {time_in_video} (index {second_index})")
                        failed_count += 1
                        continue
                
                try:
                    # Calculate rx and ry (already calculated in cartesian_path, just extract)
                    rx, ry = get_rx_ry(cartesian_point)
                    
                    # Log first frame coordinates for debugging
                    if idx == 0:
                        logger.info(f"First frame (idx=0, time={time_in_video}s): "
                                   f"cartesian_point={cartesian_point}, rx={rx}, ry={ry}, "
                                   f"starting_position={starting_position}")
                    
                    frame_id = frame.get('uuid')
                    if not frame_id:
                        logger.warning(f"Video frame missing UUID, skipping")
                        failed_count += 1
                        continue
                    
                    # Prepare batch-update entry
                    update_data.append({
                        'frame': frame_id,
                        'camera_layer_position': {
                            'rx': rx,
                            'ry': ry,
                            'layer': layer_id
                        }
                    })
                        
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f"Failed to process frame {frame.get('uuid')}: {e}")
                    failed_count += 1
                    continue
            
            # Update all frames using batch-update
            if update_data:
                logger.info(f"Updating {len(update_data)} video frames with batch-update...")
                success = self.batch_update_video_frames(update_data)
                if success:
                    updated_count = len(update_data)
                    logger.info(f"Updated {updated_count}/{len(frames)} video frames with camera positions ({failed_count} failed)")
                    if update_status_callback:
                        update_status_callback(f"Updated {updated_count} video frames with camera positions")
                    return updated_count
                else:
                    logger.error("Batch-update failed, falling back to individual updates")
                    # Fallback to individual updates
                    updated_count = 0
                    for update_entry in update_data:
                        frame_id = update_entry['frame']
                        camera_pos = update_entry['camera_layer_position']
                        if self.update_video_frame_camera_position(frame_id, camera_pos['rx'], camera_pos['ry'], layer_id):
                            updated_count += 1
                    logger.info(f"Updated {updated_count}/{len(frames)} video frames with camera positions (fallback, {failed_count} failed)")
                    if update_status_callback:
                        update_status_callback(f"Updated {updated_count} video frames with camera positions")
                    return updated_count
            else:
                logger.warning("No frames to update")
                return 0
        except Exception as e:
            logger.error(f"Error in update_video_frames_from_raw_path: {e}", exc_info=True)
            raise

    def delete_video(self, video_id: Optional[str]) -> bool:
        """Delete video asset."""
        if not video_id:
            return False
        result = self._api_request('DELETE', f'/api/v1/video/{video_id}/')
        if result is None:
            logger.warning(f"Failed to delete video {video_id}")
            return False
        logger.info(f"Deleted video {video_id}")
        return True

    def delete_video_recording(self, video_recording_id: Optional[str]) -> bool:
        """Delete video-recording object."""
        if not video_recording_id:
            return False
        result = self._api_request('DELETE', f'/api/v1/video-recording/{video_recording_id}/')
        if result is None:
            logger.warning(f"Failed to delete video-recording {video_recording_id}")
            return False
        logger.info(f"Deleted video-recording {video_recording_id}")
        return True
    
    def update_task_route(self, video_uuid: str, raw_path: Dict) -> bool:
        """Update stitched video with raw_path data."""
        if not video_uuid:
            logger.warning("Cannot update route: missing video UUID")
            return False
        
        logger.info(f"Updating video {video_uuid} with raw_path data...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/video/{video_uuid}/',
            json={'raw_path': raw_path}
        )
        return result is not None
    
    def update_video_raw_path(self, video_recording_id: str, raw_path: Dict) -> bool:
        """Update video-recording with raw_path data."""
        if not video_recording_id:
            logger.warning("Cannot update raw_path: missing video_recording_id")
            return False
        
        logger.info(f"Updating video-recording {video_recording_id} raw_path...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/video-recording/{video_recording_id}',
            json={'raw_path': raw_path}
        )
        return result is not None

    def set_video_recording_status(self, video_recording_id: str, status: str = 'created') -> bool:
        """Set video-recording status."""
        if not video_recording_id:
            logger.warning("Cannot set video-recording status: missing video_recording_id")
            return False
        
        logger.info(f"Setting video-recording {video_recording_id} status to {status}...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/video-recording/{video_recording_id}',
            json={'status': status}
        )
        return result is not None
    
    def update_video_recording_duration(self, video_recording_id: str, duration: float) -> bool:
        """Update video-recording duration in seconds."""
        if not video_recording_id:
            logger.warning("Cannot update video-recording duration: missing video_recording_id")
            return False
        
        logger.info(f"Updating video-recording {video_recording_id} duration to {duration:.2f}s...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/video-recording/{video_recording_id}',
            json={'duration': duration}
        )
        if result is not None:
            logger.info(f"Video-recording duration updated successfully")
            return True
        else:
            logger.warning(f"Failed to update video-recording duration")
            return False
    
    def update_video_recording_started(self, video_recording_id: str, recording_started: str) -> bool:
        """Update video-recording recording_started field with ISO format datetime.
        
        Args:
            video_recording_id: Video recording UUID
            recording_started: Recording start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        
        Returns:
            True if successful, False otherwise
        """
        if not video_recording_id:
            logger.warning("Cannot update recording_started: missing video_recording_id")
            return False
        
        logger.info(f"Updating video-recording {video_recording_id} recording_started to {recording_started}...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/video-recording/{video_recording_id}',
            json={'recording_started': recording_started}
        )
        if result is not None:
            logger.info(f"Video-recording recording_started updated successfully")
            return True
        else:
            logger.warning(f"Failed to update video-recording recording_started")
            return False
    
    def mark_videos_for_deletion(self, grace_period_days: int) -> bool:
        """Mark front and back videos for deletion after grace period using FileDeleteSchedule."""
        if not self.current_video_recording_id:
            logger.warning("Cannot mark videos for deletion: no current video recording ID")
            return False
        
        from datetime import datetime, timedelta
        
        # Fetch video-recording to get front and back video file UUIDs
        video_recording = self.fetch_video_recording(self.current_video_recording_id)
        if not video_recording:
            logger.error("Failed to fetch video-recording for deletion scheduling")
            return False
        
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
            return False
        
        # Get file UUIDs from videos (videos have a 'file' field that references the File object)
        # The 'file' field can be a UUID string or a dict with 'uuid' key
        front_file_uuid = front_video.get('uuid')
        back_file_uuid = back_video.get('uuid')
                
        if not front_file_uuid or not back_file_uuid:
            logger.warning(f"Missing file UUIDs in front or back video data. Front: {front_file_uuid}, Back: {back_file_uuid}")
            logger.debug(f"Front video data: {front_video}")
            logger.debug(f"Back video data: {back_video}")
            # Don't fail completely - just log warning and continue
            # The videos may have been processed differently or file references may be missing
            return False
        
        scheduled_for = datetime.now() + timedelta(days=grace_period_days)
        url = "/api/v1/file-delete-schedule/"
        
        # Create FileDeleteSchedule for front video
        front_result = self._api_request(
            'POST',
            url,
            json={
                'file': front_file_uuid,
                'scheduled_for': scheduled_for.isoformat()
            }
        )
        
        # Create FileDeleteSchedule for back video
        back_result = self._api_request(
            'POST',
            url,
            json={
                'file': back_file_uuid,
                'scheduled_for': scheduled_for.isoformat()
            }
        )
        
        if front_result and back_result:
            logger.info(f"Created FileDeleteSchedule for front video (file: {front_file_uuid}) and back video (file: {back_file_uuid})")
            return True
        else:
            logger.warning("Failed to create one or both FileDeleteSchedule objects")
            return False
        
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
                if not self.delete_image(image_id):
                    success = False
        if frame_id:
            if not self.delete_video_frame(frame_id):
                success = False
        return success
    
    def cleanup_video_recording_data(
        self,
        video_recording_id: str,
        update_status_callback,
        preserve_categories: Optional[List[str]] = None,
        test_mode: bool = False,
        test_reset_status: str = 'created'
    ) -> bool:
        """Delete videos, frames, and images associated with a video-recording."""
        video_recording = self.fetch_video_recording(video_recording_id)
        if not video_recording:
            logger.error("Reset cleanup failed: video-recording not found")
            return False

        videos = video_recording.get('videos', [])
        targets: List[Tuple[str, bool]] = []
        if preserve_categories is None:
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
            update_status_callback(f"Reset: deleting frames for video {video_id}")
            frames = self.fetch_video_frames(video_id)
            for frame in frames:
                self._delete_frame_and_images(frame)
            if delete_video_flag:
                update_status_callback(f"Reset: deleting video {video_id}")
                self.delete_video(video_id)

        if test_mode:
            # In test mode, reset status instead of deleting to avoid cascade deletion of process-recording-task
            update_status_callback("Reset: resetting video-recording status")
            self.set_video_recording_status(video_recording_id, status=test_reset_status)
            logger.info(f"Reset cleanup finished for video-recording {video_recording_id} (status reset to {test_reset_status})")
        else:
            # In normal reset mode, delete the video-recording
            update_status_callback("Reset: deleting video-recording")
            self.delete_video_recording(video_recording_id)
            logger.info(f"Reset cleanup finished for video-recording {video_recording_id}")
        return True

