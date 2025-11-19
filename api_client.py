"""
API client module for handling all API requests.
"""

import logging
import requests
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class APIClient:
    """Handles all API communication."""
    
    def __init__(self, api_domain: str, api_key: str):
        """Initialize API client."""
        self.api_domain = api_domain
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.current_task_id = None
        self.current_video_recording_id = None
    
    def _api_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling."""
        url = f"{self.api_domain}{endpoint}"
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
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    response_text = e.response.text
                    logger.error(f"API response text: {response_text}")
                except Exception:
                    pass
            return None
    
    def update_status_text(self, status_text: str) -> bool:
        """Update task status_text in API."""
        if not self.current_task_id:
            logger.warning("Cannot update status: no current task ID")
            return False
        
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}',
            json={'status_text': status_text}
        )
        return result is not None
    
    def fetch_next_task(self, reset: bool = False) -> Optional[Dict]:
        """Fetch next video_task from next endpoint."""
        logger.info("Fetching next video task...")
        url = f"/api/v1/process-recording-task/get-next-task"
        return self._api_request('GET', url, params={"reset": reset})
    
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
    
    def store_video(self, project_id: str, video_type: str, video_size: int, video_binary: bytes) -> Optional[str]:
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
                    'name': 'video',
                    'size': video_size,
                    'type': video_type
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
            payload = {
                'project': project_id,
                'video': video_id,
                'time_in_video': time_in_video,
                'high_res_image': high_res_image_id,
                'low_res_image': low_res_image_id
            }
            
            if blur_high_image_id:
                payload['blur_high_image'] = blur_high_image_id
            if blur_low_image_id:
                payload['blur_image'] = blur_low_image_id
            
            if layer_id:
                payload['camera_layer_position'] = {
                    "rx": 0,
                    "ry": 0,
                    "layer": layer_id,
                    "yaw": 0,
                    "roll": 0,
                    "pitch": 0,
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
            
            logger.info(f"Video frame saved successfully with ID: {frame_uuid}")
            return frame_uuid
            
        except Exception as e:
            logger.error(f"Failed to save video frame: {e}")
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
    
    def update_task_route(self, raw_path: Dict) -> bool:
        """Update process-recording-task with raw_path data."""
        if not self.current_task_id:
            logger.warning("Cannot update route: no current task ID")
            return False
        
        logger.info("Updating task with route data...")
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}/route',
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
    
    def mark_videos_for_deletion(self, grace_period_days: int) -> bool:
        """Mark front and back videos for deletion after grace period."""
        if not self.current_task_id:
            logger.warning("Cannot mark videos for deletion: no current task ID")
            return False
        
        from datetime import datetime, timedelta
        
        logger.info("Marking videos for deletion...")
        deletion_date = datetime.now() + timedelta(days=grace_period_days)
        result = self._api_request(
            'PATCH',
            f"/api/v1/process-recording-task/{self.current_task_id}/mark-for-deletion",
            json={
                'front_video_deletion_date': deletion_date.isoformat(),
                'back_video_deletion_date': deletion_date.isoformat()
            }
        )
        return result is not None
    
    def report_task_completion(self, success: bool, error: Optional[str] = None):
        """Report task completion to API."""
        if not self.current_task_id:
            logger.warning("Cannot report completion: no current task ID")
            return
        
        logger.info(f"Reporting task completion: success={success}")
        
        # Update process-recording-task status to "completed"
        task_result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{self.current_task_id}',
            json={'status': 'completed' if success else 'failed'}
        )
        
        # Update video-recording status to "ready_to_view" (if success)
        if success and self.current_video_recording_id:
            video_result = self._api_request(
                'PATCH',
                f'/api/v1/video-recording/{self.current_video_recording_id}',
                json={'status': 'ready_to_view'}
            )
            if video_result:
                logger.info("Video-recording status updated to ready_to_view")
            else:
                logger.warning("Failed to update video-recording status")
        elif not success:
            logger.info("Skipping video-recording status update due to failure")
        
        if task_result:
            logger.info(f"Process-recording-task status updated: {'completed' if success else 'failed'}")
        else:
            logger.warning("Failed to update process-recording-task status")

