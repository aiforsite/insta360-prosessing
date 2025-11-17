"""
Mockup test script for API integration testing.
Simulates video processing without actual video processing.
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIMockupTester:
    """Mockup tester for API integration."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize tester with configuration."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API credentials from environment
        self.api_domain = os.getenv('API_DOMAIN')
        self.api_key = os.getenv('API_KEY')
        
        if not self.api_domain or not self.api_key:
            raise ValueError("API_DOMAIN and API_KEY must be set in .env file")
        
        # Load configuration from JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.grace_period_days = self.config['video_deletion_grace_period_days']
        
        # API headers
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _api_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling."""
        url = f"{self.api_domain}{endpoint}"
        try:
            logger.info(f"API {method} {endpoint}")
            response = requests.request(
                method,
                url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            result = response.json() if response.content else {}
            logger.info(f"API response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text}")
            return None
    
    def update_status_text(self, task_id: str, status_text: str) -> bool:
        """Update task status_text in API."""
        logger.info(f"Updating status_text for task {task_id}: {status_text}")
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{task_id}',
            json={'status_text': status_text}
        )
        return result is not None
    
    def fetch_next_task(self) -> Optional[Dict]:
        """1. Fetch next video_task from next endpoint."""
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching next video task...")
        logger.info("=" * 60)
        task = self._api_request('GET', '/api/v1/process-recording-task/get-next-task?reset=true')
        if task:
            logger.info(f"✓ Task fetched: {task.get('uuid')}")
        else:
            logger.warning("✗ No task available")
        return task
    
    def simulate_step_2_cleanup(self, task_id: str):
        """2. Simulate cleaning local directories."""
        logger.info("=" * 60)
        logger.info("STEP 2: Simulating directory cleanup...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Siivotaan paikalliset hakemistot...")
        time.sleep(0.01)  # Simulate work
        logger.info("✓ Directories cleaned (simulated)")
    
    def simulate_step_3_download(self, task: Dict):
        """3. Simulate downloading videos."""
        task_id = task.get('uuid')
        logger.info("=" * 60)
        logger.info("STEP 3: Simulating video download...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Ladataan front ja back videot...")
        time.sleep(1)
        self.update_status_text(task_id, "Ladataan front video...")
        time.sleep(1)
        self.update_status_text(task_id, "Ladataan back video...")
        time.sleep(1)
        self.update_status_text(task_id, "Videot ladattu onnistuneesti")
        logger.info("✓ Videos downloaded (simulated)")
    
    def simulate_step_4_stitch(self, task_id: str):
        """4. Simulate video stitching."""
        logger.info("=" * 60)
        logger.info("STEP 4: Simulating video stitching...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Suoritetaan videoiden stitchaus...")
        time.sleep(0.01)  # Simulate longer processing
        self.update_status_text(task_id, "Stitchaus valmis")
        logger.info("✓ Videos stitched (simulated)")
    
    def simulate_step_5_frames(self, task_id: str):
        """5. Simulate frame creation."""
        logger.info("=" * 60)
        logger.info("STEP 5: Simulating frame creation...")
        logger.info("=" * 60)
        frames_per_second = self.config['frames_per_second']
        self.update_status_text(task_id, f"Luodaan framet ({frames_per_second}/s) stitchatusta tiedostosta...")
        time.sleep(0.01)
        mock_frame_count = 120  # Simulated frame count
        self.update_status_text(task_id, f"Luotu {mock_frame_count} framea")
        logger.info(f"✓ Created {mock_frame_count} frames (simulated)")
        return mock_frame_count
    
    def simulate_step_6_low_res(self, task_id: str, frame_count: int):
        """6. Simulate low res encoding."""
        logger.info("=" * 60)
        logger.info("STEP 6: Simulating low res encoding...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Enkoodataan low res framet...")
        time.sleep(0.01)
        self.update_status_text(task_id, f"Enkoodattu {frame_count} low res framea")
        logger.info(f"✓ Encoded {frame_count} low res frames (simulated)")
        return frame_count
    
    def simulate_step_7_best_frames(self, task_id: str, frame_count: int):
        """7. Simulate best frame selection."""
        logger.info("=" * 60)
        logger.info("STEP 7: Simulating best frame selection...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Poimitaan parhaat ajankohdat low res frameista (1/s intervallilla)...")
        time.sleep(0.01)
        selected_count = frame_count // self.config['low_res_frames_per_second']
        self.update_status_text(task_id, f"Valittu {selected_count} parasta framea")
        logger.info(f"✓ Selected {selected_count} best frames (simulated)")
        return selected_count
    
    def simulate_step_8_blur(self, task_id: str, frame_count: int):
        """8. Simulate frame blurring."""
        logger.info("=" * 60)
        logger.info("STEP 8: Simulating frame blurring...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Tehdään frame blurraus low res frameille...")
        time.sleep(0.01)
        self.update_status_text(task_id, f"Blurrattu {frame_count} framea")
        logger.info(f"✓ Blurred {frame_count} frames (simulated)")
        return frame_count
    
    def simulate_step_9_upload(self, task_id: str, frame_count: int):
        """9. Simulate frame upload to cloud."""
        logger.info("=" * 60)
        logger.info("STEP 9: Simulating frame upload to cloud...")
        logger.info("=" * 60)
        self.update_status_text(task_id, f"Tallennetaan {frame_count} framea pilveen ja luodaan frame objektit...")
        
        # Simulate upload progress
        for i in range(0, frame_count, 10):
            progress = min(i + 10, frame_count)
            logger.info(f"Tallennetaan frameja pilveen: {progress}/{frame_count}")
            time.sleep(0.01)
        
        logger.info(f"✓ Uploaded {frame_count} frames (simulated)")
    
    def simulate_step_10_route_frames(self, task_id: str):
        """10. Simulate route frame extraction."""
        logger.info("=" * 60)
        logger.info("STEP 10: Simulating route frame extraction...")
        logger.info("=" * 60)
        fps_range = self.config['route_calculation_fps']
        self.update_status_text(task_id, f"Poimitaan halutut framet reitin laskentaa varten ({fps_range['min']}-{fps_range['max']} frames/s)...")
        time.sleep(0.01)
        route_frame_count = 30  # Simulated
        self.update_status_text(task_id, f"Poimittu {route_frame_count} framea reitin laskentaan")
        logger.info(f"✓ Extracted {route_frame_count} route frames (simulated)")
        return route_frame_count
    
    def simulate_step_11_route_calculation(self, task_id: str, frame_count: int):
        """11. Simulate route calculation."""
        logger.info("=" * 60)
        logger.info("STEP 11: Simulating route calculation...")
        logger.info("=" * 60)
        self.update_status_text(task_id, f"Lasketaan reitti valittujen frameiden avulla ({frame_count} framea)...")
        time.sleep(0.01)  # Simulate longer processing
        
        # Create mock route data
        route_data = {
            'coordinates': [
                {'lat': 60.1699, 'lon': 24.9384, 'timestamp': time.time()},
                {'lat': 60.1700, 'lon': 24.9385, 'timestamp': time.time() + 1},
                {'lat': 60.1701, 'lon': 24.9386, 'timestamp': time.time() + 2},
            ],
            'timestamps': [time.time(), time.time() + 1, time.time() + 2],
            'distance': 150.5,
            'duration': 3.0,
            'map_db_path': '/root/stella_map.msg'  # Mock path
        }
        
        self.update_status_text(task_id, "Reitti laskettu, päivitetään raw path tulosten avulla...")
        
        # Update route via API
        result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{task_id}/route',
            json={'raw_path': route_data}
        )
        
        if result:
            logger.info("✓ Route calculated and updated via API")
        else:
            logger.warning("✗ Route calculation update failed")
        
        return route_data
    
    def simulate_step_12_deletion(self, task: Dict):
        """12. Simulate marking videos for deletion."""
        task_id = task.get('uuid')
        logger.info("=" * 60)
        logger.info("STEP 12: Simulating video deletion marking...")
        logger.info("=" * 60)
        self.update_status_text(task_id, f"Merkataan front ja back video tuhottavaksi {self.grace_period_days} päivän päästä (varoaika)...")
        time.sleep(0.01)
        
        deletion_date = datetime.now() + timedelta(days=self.grace_period_days)
        result = self._api_request(
            'PATCH',
            f"/api/v1/process-recording-task/{task_id}/mark-for-deletion",
            json={
                'front_video_deletion_date': deletion_date.isoformat(),
                'back_video_deletion_date': deletion_date.isoformat()
            }
        )
        
        if result:
            self.update_status_text(task_id, "Videot merkitty poistettavaksi")
            logger.info("✓ Videos marked for deletion")
        else:
            logger.warning("✗ Video deletion marking failed")
    
    def simulate_step_13_cleanup(self, task_id: str):
        """13. Simulate final cleanup."""
        logger.info("=" * 60)
        logger.info("STEP 13: Simulating final cleanup...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Siivotaan paikalliset hakemistot...")
        time.sleep(0.01)
        logger.info("✓ Final cleanup done (simulated)")
    
    def report_completion(self, task: Dict, success: bool, error: Optional[str] = None):
        """Report task completion to API."""
        task_id = task.get('uuid')
        logger.info("=" * 60)
        logger.info("Reporting task completion...")
        logger.info("=" * 60)
        self.update_status_text(task_id, "Tehtävä valmis, aloitetaan uuden tehtävän pollaus..." if success else f"Virhe: {error}")
        
        # Update process-recording-task status to "completed"
        task_result = self._api_request(
            'PATCH',
            f'/api/v1/process-recording-task/{task_id}',
            json={'status': 'completed' if success else 'failed'}
        )
        
        # Update video-recording status to "ready_to_view" (if success)
        video_recording_id = task.get('video_recording')
        video_result = None
        if success and video_recording_id:
            video_result = self._api_request(
                'PATCH',
                f'/api/v1/video-recording/{video_recording_id}',
                json={'status': 'ready_to_view'}
            )
        elif not success:
            logger.info("Skipping video-recording status update due to failure")
        
        if task_result:
            logger.info(f"✓ Process-recording-task status updated: {'completed' if success else 'failed'}")
        else:
            logger.warning("✗ Process-recording-task status update failed")
        
        if video_result:
            logger.info("✓ Video-recording status updated: ready_to_view")
        elif success and video_recording_id:
            logger.warning("✗ Video-recording status update failed")
    
    def run_mockup_test(self):
        """Run complete mockup test."""
        logger.info("=" * 60)
        logger.info("STARTING API MOCKUP TEST")
        logger.info("=" * 60)
        logger.info(f"API Domain: {self.api_domain}")
        logger.info("")
        
        # Step 1: Fetch task
        task = self.fetch_next_task()
        
        if not task or not task.get('uuid'):
            logger.warning("No task available for testing. Exiting.")
            return False
        
        task_id = task.get('uuid')
        logger.info(f"Processing mockup test for task: {task_id}")
        logger.info("")
        
        try:
            # Step 2: Cleanup
            self.simulate_step_2_cleanup(task_id)
            time.sleep(0.01)
            
            # Step 3: Download
            self.simulate_step_3_download(task)
            time.sleep(0.01)
            
            # Step 4: Stitch
            self.simulate_step_4_stitch(task_id)
            time.sleep(0.01)
            
            # Step 5: Frames
            frame_count = self.simulate_step_5_frames(task_id)
            time.sleep(0.01)
            
            # Step 6: Low res
            low_res_count = self.simulate_step_6_low_res(task_id, frame_count)
            time.sleep(0.01)
            
            # Step 7: Best frames
            best_count = self.simulate_step_7_best_frames(task_id, low_res_count)
            time.sleep(0.01)
            
            # Step 8: Blur
            blurred_count = self.simulate_step_8_blur(task_id, best_count)
            time.sleep(0.01)
            
            # Step 9: Upload
            self.simulate_step_9_upload(task_id, blurred_count)
            time.sleep(0.01)
            
            # Step 10: Route frames
            route_frame_count = self.simulate_step_10_route_frames(task_id)
            time.sleep(0.01)
            
            # Step 11: Route calculation
            route_data = self.simulate_step_11_route_calculation(task_id, route_frame_count)
            time.sleep(0.01)
            
            # Step 12: Deletion
            self.simulate_step_12_deletion(task)
            time.sleep(0.01)
            
            # Step 13: Final cleanup
            self.simulate_step_13_cleanup(task_id)
            time.sleep(0.01)
            
            # Report success
            self.report_completion(task, success=True)
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("MOCKUP TEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Mockup test failed: {e}")
            self.report_completion(task, success=False, error=str(e))
            return False


if __name__ == "__main__":
    tester = APIMockupTester()
    tester.run_mockup_test()

