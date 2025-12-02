"""
API client module for handling all API requests.
"""

import logging
import math
import time
import requests
from typing import Dict, Optional, List, Any, Tuple

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
        logger.info("Fetching next video task from Media Server API...")
        payload = {
            "task_type": "process_360_video", 
            "environment": "development", # production
            "worker_id": self.worker_id,
            "change_status": "false" if reset else "true"
        }
        endpoint = f"/api/tasks/fetch"
        task = self._api_request('POST', endpoint, json=payload)
        if task:
            logger.info(f"Fetched task: {task}")
            return task
        else:
            logger.info("No task found")
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
                error_msg = f"API request failed (attempt {attempt + 1}/{max_retries}): {e}"
                if is_last_attempt:
                    logger.error(error_msg)
                else:
                    logger.warning(error_msg)
                
                # Log response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        response_text = e.response.text
                        status_code = e.response.status_code
                        if is_last_attempt:
                            logger.error(f"API response status: {status_code}, text: {response_text[:500]}")
                        else:
                            logger.warning(f"API response status: {status_code}, text: {response_text[:200]}")
                    except Exception:
                        pass
                
                # Don't retry on last attempt
                if is_last_attempt:
                    return None
                
                # Calculate exponential backoff delay: 1s, 2s, 4s, etc.
                delay = 2 ** attempt
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Should not reach here, but just in case
        return None
    
    def fetch_video_recording(self, video_recording_id: str) -> Optional[Dict]:
        """Fetch video-recording data from API."""
        logger.info(f"Fetching video-recording data for {video_recording_id}...")
        return self._api_request('GET', f'/api/v1/video-recording/{video_recording_id}/')
    
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
            
            # Step 2: Upload video binary to upload_url
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
            
            # Step 3: Register file with API
            logger.info(f"Registering file for video {video_uuid}...")
            register_response = self._api_request(
                'PUT',
                f'/api/v1/video/{video_uuid}/register_file',
            )
            
            if not register_response:
                logger.error("Failed to register file with API")
                return None
            
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

    def fetch_video_frames(self, video_id: str, page_size: int = 200) -> List[Dict]:
        """Fetch all video-frame records for a given video."""
        frames: List[Dict] = []
        page = 1
        while True:
            params = {'video': video_id, 'page': page, 'page_size': page_size}
            response = self._api_request('GET', '/api/v1/video-frame/', params=params)
            if not response:
                break
            if isinstance(response, dict) and 'results' in response:
                results = response.get('results') or []
                frames.extend(results)
                if response.get('next'):
                    page += 1
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
        front_file_uuid = front_video.get('file')
        back_file_uuid = back_video.get('file')
        
        if not front_file_uuid or not back_file_uuid:
            logger.error("Missing file UUIDs in front or back video data")
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

