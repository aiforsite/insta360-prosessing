# Insta360 Processing

Project to process Insta360 material (videos and images) in one pipeline.

## Overview

This project contains a video processing script that:
1. Fetches video processing tasks from an API
2. Downloads front and back videos
3. Stitches videos together
4. Extracts and processes frames
5. Calculates routes from video data
6. Uploads results to cloud storage
7. Manages video lifecycle (deletion scheduling)

## Setup

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- API access to the target system
- PyTorch + TorchVision (for GPU-accelerated blur, optional but recommended)

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

3. Configure `config.json`:
   - Set API credentials (`api_domain`, `api_key`)
   - Set MediaSDK executable path (`mediasdk_executable`) - Windows path to MediaSDKTest.exe
   - Set Stella VSLAM paths (`stella_executable`, `stella_config_path`, `stella_vocab_path`) - optional if `enable_stella` is false
   - Set `enable_stella` to `true` or `false` to enable/disable route calculation
   - Set `stella_frames_category` (defaults to `general_zip`) for the storage category used when uploading the Stella frame ZIP (used when `enable_stella` is false)
   - Adjust processing parameters as needed (polling interval, FPS, etc.)

## Configuration

### Application Configuration (config.json)

Edit `config.json` to configure processing parameters:

```json
{
  "api_domain": "https://your-api-domain.com",
  "mediasdk_executable": "C:\\Users\\Administrator\\Documents\\insta360\\MediaSDK\\bin\\MediaSDKTest.exe",
  "enable_stella": false,
  "stella_frames_category": "general_zip",
  "api_key": "your-api-key-here",
  "media_model_dir": "/path/to/media/model/directory",
  "stella_executable": "/root/lib/stella_vslam_examples/build/run_image_slam",
  "stella_config_path": "/opt/stella_vslam/insta360_equirect.yaml",
  "stella_vocab_path": "/opt/stella_vslam/orb_vocab.fbow",
  "polling_interval": 15,
  "local_work_dir": "./work",
  "frames_per_second": 12,
  "low_res_frames_per_second": 1,
  "candidates_per_second": 12,
  "stella_fps": 12,
  "route_calculation_fps": {
    "min": 1,
    "max": 4
  },
  "video_deletion_grace_period_days": 30,
  "blur_settings": {
    "enable_gpu": true,
    "confidence_threshold": 0.7,
    "min_box_area": 2500,
    "blur_radius": 18,
    "fallback_full_frame": true
  },
  "fallback_video_categories": [
    "video_insta360_raw_front",
    "video_insta360_raw_back"
  ],
  "test_task_uuid": ""
}

#### Blur Settings

- `enable_gpu`: Toggle GPU-accelerated person detection (requires PyTorch + TorchVision).
- `confidence_threshold`: Minimum detection confidence (0–1) to treat a box as a person.
- `min_box_area`: Ignore detections smaller than this pixel area (filters out noise).
- `blur_radius`: Gaussian blur radius applied to each bounding box.
- `fallback_full_frame`: When `true`, fall back to full-frame blur if no people are detected or the detector is unavailable.
- `test_task_uuid`: UUID of a process-recording-task used when running `processing_runner.py --test`.
```

> **Security note:** `config.json` now contains API keys and other secrets. Keep it out of version control or manage environment-specific copies securely.

- `fallback_video_categories`: Preference order for existing videos to reuse if stitched upload fails.

## Usage

### Testing

#### Mockup Test (API Integration Only)

Test API integration without actual video processing:

```bash
python test_api_mockup.py
```

This will:
- Fetch a task from the API
- Simulate all processing steps without actual video processing
- Test all API endpoints (status updates, route updates, completion, etc.)
- Help verify API connectivity and endpoint correctness

#### Component Tests (Using Real VideoProcessor Class)

Test individual components using the actual VideoProcessor class:

```bash
python test_processing_runner.py
```

Or run component tests only:
```bash
python test_processing_runner.py --component
```

This will test:
- API connection
- Video recording data fetch
- Image storage (`api_client.store_image`)
- Video storage (`api_client.store_video`)
- Video frame creation (`api_client.save_videoframe`)
- Status updates
- Sharpness calculation

#### Full Task Processing Test

Test the complete processing pipeline with real videos:

```bash
python test_processing_runner.py --full
```

**Warning**: This will process actual videos and may take a long time. Use with caution!

### Running the Full Video Processor

Run the video processor:

```bash
python processing_runner.py
```

The script will:
- Poll the API every 15 seconds (configurable) for new tasks
- Process each task through the complete pipeline
- Report results back to the API
- Clean up local files after processing

#### Reset Mode

Use reset mode to clean up API data (videos, frames, linked images) for queued tasks:

```bash
python processing_runner.py --reset
```

This will:
- Fetch reset tasks via `?reset=true`
- Delete all video frames and their associated images
- Delete related video assets
- Delete the video-recording entry after cleanup

#### Test Mode

Use test mode to run a single reset iteration against a predefined task (configure `test_task_uuid` in `config.json`):

```bash
python processing_runner.py --test
```

This will:
- Patch the configured task back to `pending`
- Fetch it with `reset=true`
- Run the cleanup workflow exactly once (no polling loop)

## Processing Pipeline

1. **Fetch Task**: Polls API for next video task
2. **Clean Directories**: Removes old work files
3. **Download Videos**: Downloads front and back videos
4. **Stitch Videos**: Combines videos into single output
5. **Extract Frames**: Creates frames at 12-15 FPS
6. **Encode Low Res**: Creates low resolution versions
7. **Select Best Frames**: Picks best frames at 1/s interval
8. **Blur Frames**: Applies blur to low res frames
9. **Upload to Cloud**: Saves frames and creates frame objects
10. **Extract Route Frames**: Gets frames for route calculation (1-4 FPS)
11. **Calculate Route**: Runs Stella VSLAM on extracted frames and uploads results
12. **Mark for Deletion**: Schedules video deletion after grace period
13. **Cleanup**: Removes local files and continues polling

## API Endpoints

The script expects the following API endpoints:

- `GET /api/video-tasks/next` - Fetch next task
- `PATCH /api/video-tasks/{id}/status` - Update task status_text
- `POST /api/frames/upload` - Upload frame to cloud
- `PATCH /api/video-tasks/{id}/route` - Update task with route data
- `PATCH /api/video-tasks/{id}/mark-for-deletion` - Schedule video deletion
- `PATCH /api/video-tasks/{id}/complete` - Report task completion

## Status Reporting

The script reports progress to the API using the `status_text` field at each processing step:
- Directory cleanup
- Video downloading
- Video stitching
- Frame extraction
- Low resolution encoding
- Best frame selection
- Frame blurring
- Cloud upload
- Route calculation
- Video deletion marking
- Final cleanup

## Notes

- MediaSDKTest is used for video stitching (requires MediaSDK installation)
- FFmpeg is used for frame extraction
- Stella VSLAM (`run_image_slam`) is used for route calculation and produces `stella_map.msg`
- Local work directory is cleaned before and after each task
- Videos are marked for deletion 30 days after processing (configurable)
- All API requests use Bearer token authentication
- API credentials and tool paths are read from `config.json`

## Development

### TODO Items

Some functions need implementation:
- `stitch_videos()`: Implement actual video stitching logic
- `calculate_route()`: Implement GPS/route extraction from frames
- Frame quality analysis for best frame selection
- Cloud storage integration details

## License

[Add your license here]
