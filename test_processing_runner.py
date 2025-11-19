"""
Test script for VideoProcessor class.
Tests actual functionality using real API calls.
"""

import os
import sys
import logging
from pathlib import Path
from processing_runner import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_connection(processor: VideoProcessor):
    """Test basic API connection."""
    logger.info("=" * 60)
    logger.info("TEST 1: API Connection Test")
    logger.info("=" * 60)
    
    try:
        task = processor.api_client.fetch_next_task()
        if task:
            logger.info(f"✓ API connection successful. Task fetched: {task.get('uuid')}")
            return task
        else:
            logger.warning("✗ No task available, but API connection works")
            return None
    except Exception as e:
        logger.error(f"✗ API connection failed: {e}")
        return None


def test_video_recording_fetch(processor: VideoProcessor, video_recording_id: str):
    """Test fetching video-recording data."""
    logger.info("=" * 60)
    logger.info("TEST 2: Video Recording Fetch Test")
    logger.info("=" * 60)
    
    try:
        video_recording = processor.api_client.fetch_video_recording(video_recording_id)
        if video_recording:
            logger.info(f"✓ Video recording fetched successfully")
            logger.info(f"  - UUID: {video_recording.get('uuid')}")
            logger.info(f"  - Project: {video_recording.get('project')}")
            logger.info(f"  - Videos count: {len(video_recording.get('videos', []))}")
            return video_recording
        else:
            logger.error("✗ Failed to fetch video recording")
            return None
    except Exception as e:
        logger.error(f"✗ Video recording fetch failed: {e}")
        return None


def test_store_image(processor: VideoProcessor, project_id: str):
    """Test storing an image."""
    logger.info("=" * 60)
    logger.info("TEST 3: Store Image Test")
    logger.info("=" * 60)
    
    try:
        # Create a small test image (1x1 pixel PNG)
        test_image_binary = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        image_id = processor.api_client.store_image(
            project_id=project_id,
            image_type='test',
            image_size=len(test_image_binary),
            image_binary=test_image_binary
        )
        
        if image_id:
            logger.info(f"✓ Image stored successfully with ID: {image_id}")
            return image_id
        else:
            logger.error("✗ Failed to store image")
            return None
    except Exception as e:
        logger.error(f"✗ Store image failed: {e}")
        return None


def test_store_video(processor: VideoProcessor, project_id: str):
    """Test storing a video."""
    logger.info("=" * 60)
    logger.info("TEST 4: Store Video Test")
    logger.info("=" * 60)
    
    try:
        # Create a minimal test video binary (just a placeholder)
        # In real scenario, this would be actual video data
        test_video_binary = b'test video binary data'
        
        video_id = processor.api_client.store_video(
            project_id=project_id,
            video_type='test',
            video_size=len(test_video_binary),
            video_binary=test_video_binary
        )
        
        if video_id:
            logger.info(f"✓ Video stored successfully with ID: {video_id}")
            return video_id
        else:
            logger.error("✗ Failed to store video")
            return None
    except Exception as e:
        logger.error(f"✗ Store video failed: {e}")
        return None


def test_save_videoframe(processor: VideoProcessor, project_id: str, video_id: str, high_res_image_id: str, low_res_image_id: str):
    """Test saving a video frame."""
    logger.info("=" * 60)
    logger.info("TEST 5: Save Video Frame Test")
    logger.info("=" * 60)
    
    try:
        frame_uuid = processor.api_client.save_videoframe(
            project_id=project_id,
            video_id=video_id,
            time_in_video=1.5,  # 1.5 seconds
            high_res_image_id=high_res_image_id,
            low_res_image_id=low_res_image_id
        )
        
        if frame_uuid:
            logger.info(f"✓ Video frame saved successfully with ID: {frame_uuid}")
            return frame_uuid
        else:
            logger.error("✗ Failed to save video frame")
            return None
    except Exception as e:
        logger.error(f"✗ Save video frame failed: {e}")
        return None


def test_status_update(processor: VideoProcessor, task_id: str):
    """Test status text update."""
    logger.info("=" * 60)
    logger.info("TEST 6: Status Update Test")
    logger.info("=" * 60)
    
    try:
        processor.api_client.current_task_id = task_id
        result = processor.update_status_text("Test status update from test script")
        
        if result:
            logger.info("✓ Status update successful")
            return True
        else:
            logger.error("✗ Status update failed")
            return False
    except Exception as e:
        logger.error(f"✗ Status update failed: {e}")
        return False


def test_sharpness_calculation(processor: VideoProcessor):
    """Test sharpness calculation if test image exists."""
    logger.info("=" * 60)
    logger.info("TEST 7: Sharpness Calculation Test")
    logger.info("=" * 60)
    
    try:
        # Try to find a test image in work directory
        work_dir = Path(processor.config.get('local_work_dir', './work'))
        test_images = list(work_dir.rglob("*.jpg")) + list(work_dir.rglob("*.png"))
        
        if not test_images:
            logger.warning("No test images found, skipping sharpness test")
            return None
        
        test_image = test_images[0]
        logger.info(f"Testing sharpness calculation on: {test_image}")
        sharpness = processor._calculate_sharpness(test_image)
        logger.info(f"✓ Sharpness calculated: {sharpness:.2f}")
        return sharpness
    except Exception as e:
        logger.error(f"✗ Sharpness calculation failed: {e}")
        return None


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("STARTING VIDEO PROCESSOR TESTS")
    logger.info("=" * 60)
    
    try:
        # Initialize processor
        processor = VideoProcessor()
        logger.info(f"Processor initialized")
        logger.info(f"API Domain: {processor.api_domain}")
        logger.info("")
        
        # Test 1: API Connection
        task = test_api_connection(processor)
        if not task:
            logger.warning("No task available for further testing")
            return
        
        task_id = task.get('uuid')
        video_recording_id = task.get('video_recording')
        
        if not video_recording_id:
            logger.warning("No video_recording_id in task, skipping some tests")
            return
        
        # Test 2: Video Recording Fetch
        video_recording = test_video_recording_fetch(processor, video_recording_id)
        if not video_recording:
            logger.error("Cannot continue without video recording data")
            return
        
        project_id = video_recording.get('project')
        logger.info("")
        
        # Test 3: Status Update
        test_status_update(processor, task_id)
        logger.info("")
        
        # Test 4: Store Image
        high_res_image_id = test_store_image(processor, project_id)
        logger.info("")
        
        # Test 5: Store Image (low res)
        low_res_image_id = test_store_image(processor, project_id)
        logger.info("")
        
        # Test 6: Store Video (if we have image IDs)
        if high_res_image_id and low_res_image_id:
            video_id = test_store_video(processor, project_id)
            logger.info("")
            
            # Test 7: Save Video Frame
            if video_id:
                test_save_videoframe(processor, project_id, video_id, high_res_image_id, low_res_image_id)
                logger.info("")
        
        # Test 8: Sharpness Calculation (if images exist)
        test_sharpness_calculation(processor)
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()


def run_single_task_test():
    """Run a single task through the full processing pipeline."""
    logger.info("=" * 60)
    logger.info("FULL TASK PROCESSING TEST")
    logger.info("=" * 60)
    
    try:
        processor = VideoProcessor()
        logger.info(f"Processor initialized")
        logger.info(f"API Domain: {processor.api_domain}")
        logger.info("")
        
        # Fetch next task
        task = processor.api_client.fetch_next_task(reset=True)
        if not task or not task.get('uuid'):
            logger.warning("No task available for processing")
            return
        
        logger.info(f"Found task: {task.get('uuid')}")
        logger.info("Starting full task processing...")
        logger.info("")
        
        # Process the task
        success = processor.process_task(task)
        
        if success:
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ TASK PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
        else:
            logger.error("")
            logger.error("=" * 60)
            logger.error("✗ TASK PROCESSING FAILED")
            logger.error("=" * 60)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Full task test failed: {e}")
        import traceback
        traceback.print_exc()


def test_storage_methods(processor: VideoProcessor, project_id: str):
    """Test only store_image, store_video, and save_videoframe methods."""
    logger.info("=" * 60)
    logger.info("STORAGE METHODS TEST")
    logger.info("=" * 60)
    logger.info("Testing: store_image, store_video, save_videoframe")
    logger.info("")
    
    try:
        # Test 1: Store Image (high res)
        logger.info("=" * 60)
        logger.info("TEST 1: Store Image (high res)")
        logger.info("=" * 60)
        test_image_binary = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        high_res_image_id = processor.api_client.store_image(
            project_id=project_id,
            image_type='high',
            image_size=len(test_image_binary),
            image_binary=test_image_binary
        )
        
        if not high_res_image_id:
            logger.error("✗ Failed to store high res image")
            return False
        
        logger.info(f"✓ High res image stored: {high_res_image_id}")
        logger.info("")
        
        # Test 2: Store Image (low res)
        logger.info("=" * 60)
        logger.info("TEST 2: Store Image (low res)")
        logger.info("=" * 60)
        low_res_image_id = processor.api_client.store_image(
            project_id=project_id,
            image_type='low',
            image_size=len(test_image_binary),
            image_binary=test_image_binary
        )
        
        if not low_res_image_id:
            logger.error("✗ Failed to store low res image")
            return False
        
        logger.info(f"✓ Low res image stored: {low_res_image_id}")
        logger.info("")
        
        # Test 3: Store Video
        logger.info("=" * 60)
        logger.info("TEST 3: Store Video")
        logger.info("=" * 60)
        test_video_binary = b'test video binary data for testing'
        
        video_id = processor.api_client.store_video(
            project_id=project_id,
            video_type='stitched',
            video_size=len(test_video_binary),
            video_binary=test_video_binary
        )
        
        if not video_id:
            logger.error("✗ Failed to store video")
            return False
        
        logger.info(f"✓ Video stored: {video_id}")
        logger.info("")
        
        # Test 4: Save Video Frame
        logger.info("=" * 60)
        logger.info("TEST 4: Save Video Frame")
        logger.info("=" * 60)
        frame_uuid = processor.api_client.save_videoframe(
            project_id=project_id,
            video_id=video_id,
            time_in_video=1.5,  # 1.5 seconds
            high_res_image_id=high_res_image_id,
            low_res_image_id=low_res_image_id
        )
        
        if not frame_uuid:
            logger.error("✗ Failed to save video frame")
            return False
        
        logger.info(f"✓ Video frame saved: {frame_uuid}")
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("✓ ALL STORAGE METHODS TESTS PASSED")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"✗ Storage methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test VideoProcessor functionality')
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full task processing test (processes actual videos)'
    )
    parser.add_argument(
        '--component',
        action='store_true',
        help='Run component tests only (API, storage, etc.)'
    )
    parser.add_argument(
        '--storage',
        action='store_true',
        help='Test only storage methods: store_image, store_video, save_videoframe'
    )
    
    args = parser.parse_args()
    
    if args.full:
        run_single_task_test()
    elif args.storage:
        # Test only storage methods
        logger.info("Initializing VideoProcessor...")
        try:
            processor = VideoProcessor()
            logger.info(f"API Domain: {processor.api_domain}")
            logger.info("")
            
            # Get project_id from a task or video_recording
            task = processor.api_client.fetch_next_task()
            if not task:
                logger.error("No task available. Cannot get project_id.")
                sys.exit(1)
            
            video_recording_id = task.get('video_recording')
            if not video_recording_id:
                logger.error("No video_recording_id in task. Cannot get project_id.")
                sys.exit(1)
            
            video_recording = processor.api_client.fetch_video_recording(video_recording_id)
            if not video_recording:
                logger.error("Failed to fetch video recording. Cannot get project_id.")
                sys.exit(1)
            
            project_id = video_recording.get('project')
            if not project_id:
                logger.error("No project_id in video recording.")
                sys.exit(1)
            
            logger.info(f"Using project_id: {project_id}")
            logger.info("")
            
            # Run storage tests
            success = test_storage_methods(processor, project_id)
            sys.exit(0 if success else 1)
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        run_all_tests()

