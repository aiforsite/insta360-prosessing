"""
Mock tests for processing_runner.py
Tests the logic without requiring real API, files, or external dependencies.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from pathlib import Path
import json
import tempfile
import shutil

from processing_runner import VideoProcessor


class TestVideoProcessorMock(unittest.TestCase):
    """Mock tests for VideoProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for work
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'api_domain': 'http://test-api.example.com',
            'api_key': 'test-api-key',
            'media_server_api_key': 'test-media-key',
            'media_server_api_domain': 'http://test-media.example.com',
            'worker_id': 'test_worker_1',
            'mediasdk_executable': 'test_mediasdk.exe',
            'stella_executable': 'test_stella',
            'stella_config_path': '/test/stella_config.yaml',
            'stella_vocab_path': '/test/stella_vocab.fbow',
            'stella_results_path': './stella_results',
            'stella_use_wsl': False,
            'stella_use_docker': False,
            'polling_interval': 1,
            'local_work_dir': str(self.test_dir),
            'frames_per_second': 12,
            'low_res_frames_per_second': 1,
            'route_calculation_fps': {'min': 1, 'max': 4},
            'candidates_per_second': 12,
            'video_deletion_grace_period_days': 30,
            'blur_settings': {},
            'stitch_config': {},
            'fallback_video_categories': ['video_insta360_raw_front', 'video_insta360_raw_back'],
            'processed_video_category': 'video_insta360_processed_stitched',
            'test_video_recording_reset_status': 'created'
        }
        
        # Create config file
        self.config_path = self.test_dir / 'test_config.json'
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_init(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor(str(self.config_path))
        
        # Verify config loaded
        self.assertEqual(processor.config, self.config)
        self.assertEqual(processor.work_dir, self.test_dir)
        
        # Verify modules initialized
        mock_api.assert_called_once()
        mock_media_api.assert_called_once()
        mock_file.assert_called_once()
        mock_video.assert_called_once()
        mock_frame.assert_called_once()
        mock_route.assert_called_once()
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_update_api_client_from_task(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test API client update from task."""
        processor = VideoProcessor(str(self.config_path))
        
        # Mock API client
        mock_api_instance = Mock()
        mock_api_instance.api_domain = 'http://old-api.example.com'
        mock_api_instance.api_key = 'old-key'
        processor.api_client = mock_api_instance
        
        # Test with task containing domain and token
        task = {
            'uuid': 'test-task-1',
            'domain': 'http://new-api.example.com',
            'token': 'new-token'
        }
        
        processor._update_api_client_from_task(task)
        
        # Verify update_credentials was called
        mock_api_instance.update_credentials.assert_called_once_with(
            'http://new-api.example.com',
            'new-token'
        )
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_update_api_client_from_task_fallback_to_config(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test API client uses config when task doesn't provide credentials."""
        processor = VideoProcessor(str(self.config_path))
        
        # Mock API client with config values
        mock_api_instance = Mock()
        mock_api_instance.api_domain = 'http://config-api.example.com'
        mock_api_instance.api_key = 'config-key'
        processor.api_client = mock_api_instance
        
        # Task without domain/token
        task = {
            'uuid': 'test-task-1'
        }
        
        # Should not raise error if config has values
        processor._update_api_client_from_task(task)
        
        # update_credentials should not be called
        mock_api_instance.update_credentials.assert_not_called()
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_process_task_full_flow(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test full process_task flow with mocks."""
        processor = VideoProcessor(str(self.config_path))
        
        # Setup mocks
        mock_api_instance = Mock()
        mock_api_instance.current_task_id = 'test-task-1'
        mock_api_instance.current_video_recording_id = 'test-recording-1'
        mock_api_instance.test_mode = False
        mock_api_instance.fetch_video_recording.return_value = {
            'uuid': 'test-recording-1',
            'project': 'test-project-1',
            'layer': 'test-layer-1',
            'blur_people': False,
            'videos': []
        }
        mock_api_instance.set_video_recording_status.return_value = True
        mock_api_instance.update_task_route.return_value = True
        mock_api_instance.update_video_recording_duration.return_value = True
        mock_api_instance.update_video_frames_from_raw_path.return_value = 10
        mock_api_instance.mark_videos_for_deletion.return_value = True
        processor.api_client = mock_api_instance
        
        mock_media_api_instance = Mock()
        mock_media_api_instance.update_task_status.return_value = True
        processor.media_server_api_client = mock_media_api_instance
        processor.update_status_text = mock_media_api_instance.update_task_status
        
        # Mock video processing
        mock_video_instance = Mock()
        front_path = self.test_dir / 'front.mp4'
        back_path = self.test_dir / 'back.mp4'
        front_path.touch()
        back_path.touch()
        mock_video_instance.download_videos.return_value = (front_path, back_path)
        mock_video_instance.stitch_videos.return_value = True
        mock_video_instance.store_processed_video.return_value = 'test-video-id-1'
        mock_video_instance.select_fallback_video_id.return_value = None
        processor.video_processing = mock_video_instance
        
        # Mock frame processing
        mock_frame_instance = Mock()
        frame1 = self.test_dir / 'frame1.jpg'
        frame1_low = self.test_dir / 'frame1_low.jpg'
        stella_frame1 = self.test_dir / 'stella_frame1.jpg'
        frame1.touch()
        frame1_low.touch()
        stella_frame1.touch()
        mock_frame_instance.create_and_select_frames.return_value = (
            [frame1],
            [frame1_low]
        )
        mock_frame_instance.blur_frames_optional.return_value = (
            [frame1],
            [frame1_low]
        )
        mock_frame_instance.upload_frames_to_cloud.return_value = [{'uuid': 'frame-1'}]
        mock_frame_instance.create_stella_frames.return_value = [stella_frame1]
        processor.frame_processing = mock_frame_instance
        
        # Mock route calculation
        mock_route_instance = Mock()
        stitched_path = self.test_dir / 'stitched_output.mp4'
        stitched_path.touch()
        stitched_path.write_bytes(b'fake video data')
        mock_route_instance.calculate_route.return_value = {
            'raw_path': {'0': [0.0, '0.0 1.0 2.0 3.0 0 0 0 1']},
            'duration': 10.0
        }
        processor.route_calculation = mock_route_instance
        
        # Mock file operations
        mock_file_instance = Mock()
        processor.file_ops = mock_file_instance
        
        # Test task (using correct structure)
        task = {
            'task_id': 'test-task-1',
            'details': {
                'video_recording': 'test-recording-1'
            },
            'domain': 'http://test-api.example.com',
            'token': 'test-token'
        }
        
        # Execute
        result = processor.process_task(task)
        
        # Verify
        self.assertTrue(result)
        mock_api_instance.update_credentials.assert_called_once()
        mock_video_instance.download_videos.assert_called_once()
        mock_video_instance.stitch_videos.assert_called_once()
        mock_frame_instance.create_and_select_frames.assert_called_once()
        mock_frame_instance.blur_frames_optional.assert_called_once()
        mock_frame_instance.upload_frames_to_cloud.assert_called_once()
        mock_frame_instance.create_stella_frames.assert_called_once()
        mock_route_instance.calculate_route.assert_called_once()
        # Verify video-recording status was set to ready_to_view on success
        mock_api_instance.set_video_recording_status.assert_any_call('test-recording-1', status='ready_to_view')
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_process_task_error_handling(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test process_task error handling."""
        processor = VideoProcessor(str(self.config_path))
        
        # Setup mocks
        mock_api_instance = Mock()
        mock_api_instance.current_task_id = 'test-task-1'
        mock_api_instance.current_video_recording_id = 'test-recording-1'
        mock_api_instance.test_mode = False
        # First call succeeds (for set_video_recording_status('processing')), second fails
        mock_api_instance.fetch_video_recording.side_effect = [
            {'uuid': 'test-recording-1', 'project': 'test-project-1', 'videos': []},  # First call
            None  # Second call fails
        ]
        mock_api_instance.set_video_recording_status.return_value = True
        processor.api_client = mock_api_instance
        
        mock_media_api_instance = Mock()
        mock_media_api_instance.update_task_status.return_value = True
        processor.media_server_api_client = mock_media_api_instance
        processor.update_status_text = mock_media_api_instance.update_task_status
        
        mock_file_instance = Mock()
        processor.file_ops = mock_file_instance
        
        # Test task (using correct structure)
        task = {
            'task_id': 'test-task-1',
            'details': {
                'video_recording': 'test-recording-1'
            },
            'domain': 'http://test-api.example.com',
            'token': 'test-token'
        }
        
        # Execute
        result = processor.process_task(task)
        
        # Verify error handling
        self.assertFalse(result)
        # Verify video-recording status was set to failure on error
        mock_api_instance.set_video_recording_status.assert_any_call('test-recording-1', status='failure')
        mock_file_instance.clean_local_directories.assert_called()
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_handle_reset_task(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test handle_reset_task with mocks."""
        processor = VideoProcessor(str(self.config_path))
        
        # Setup mocks
        mock_api_instance = Mock()
        mock_api_instance.current_task_id = 'test-reset-task-1'
        mock_api_instance.current_video_recording_id = 'test-recording-1'
        mock_api_instance.cleanup_video_recording_data.return_value = True
        mock_api_instance.set_video_recording_status.return_value = True
        processor.api_client = mock_api_instance
        
        mock_media_api_instance = Mock()
        mock_media_api_instance.update_task_status.return_value = True
        processor.media_server_api_client = mock_media_api_instance
        processor.update_status_text = mock_media_api_instance.update_task_status
        
        # Test task
        task = {
            'task_id': 'test-reset-task-1',
            'video_recording': 'test-recording-1',
            'domain': 'http://test-api.example.com',
            'token': 'test-token'
        }
        
        # Execute
        result = processor.handle_reset_task(task, test_mode=False)
        
        # Verify
        self.assertTrue(result)
        mock_api_instance.update_credentials.assert_called_once()
        mock_api_instance.cleanup_video_recording_data.assert_called_once()
        # Verify video-recording status was set to completed on success (when test_mode=False)
        mock_api_instance.set_video_recording_status.assert_called_once_with('test-recording-1', status='completed')
    
    @patch('processing_runner.APIClient')
    @patch('processing_runner.MediaServerAPIClient')
    @patch('processing_runner.FileOperations')
    @patch('processing_runner.VideoProcessing')
    @patch('processing_runner.FrameProcessing')
    @patch('processing_runner.RouteCalculation')
    def test_clean_local_directories(self, mock_route, mock_frame, mock_video, mock_file, mock_media_api, mock_api):
        """Test clean_local_directories."""
        processor = VideoProcessor(str(self.config_path))
        
        # Setup mocks
        mock_api_instance = Mock()
        mock_api_instance.current_task_id = 'test-task-1'
        processor.api_client = mock_api_instance
        
        mock_media_api_instance = Mock()
        mock_media_api_instance.update_task_status.return_value = True
        processor.media_server_api_client = mock_media_api_instance
        processor.update_status_text = mock_media_api_instance.update_task_status
        
        mock_file_instance = Mock()
        processor.file_ops = mock_file_instance
        
        # Execute
        processor.clean_local_directories()
        
        # Verify
        mock_file_instance.clean_local_directories.assert_called_once()
        # Verify status update was called (may be called with different parameters)
        self.assertTrue(mock_media_api_instance.update_task_status.called)


if __name__ == '__main__':
    unittest.main()

