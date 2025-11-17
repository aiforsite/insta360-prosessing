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

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

3. Configure the environment:
   - Copy `.env.example` to `.env`
   - Edit `.env` and set:
     - `API_DOMAIN`: Your API domain URL
     - `API_KEY`: Your API authentication key
     - `MEDIA_MODEL_DIR`: Path to MediaSDK model directory (e.g., `/path/to/models`)
     - `STELLA_EXECUTABLE`: Path to `run_image_slam` binary
     - `STELLA_CONFIG_PATH`: Path to `insta360_equirect.yaml`
     - `STELLA_VOCAB_PATH`: Path to `orb_vocab.fbow`

## Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```env
API_DOMAIN=https://your-api-domain.com
API_KEY=your-api-key-here
MEDIA_MODEL_DIR=/path/to/media/model/directory
STELLA_EXECUTABLE=/root/lib/stella_vslam_examples/build/run_image_slam
STELLA_CONFIG_PATH=/opt/stella_vslam/insta360_equirect.yaml
STELLA_VOCAB_PATH=/opt/stella_vslam/orb_vocab.fbow
```

### Application Configuration (config.json)

Edit `config.json` to configure processing parameters:

```json
{
  "polling_interval": 15,
  "local_work_dir": "./work",
  "frames_per_second": 12,
  "low_res_frames_per_second": 1,
  "route_calculation_fps": {
    "min": 1,
    "max": 4
  },
  "video_deletion_grace_period_days": 30
}
```

## Usage

### Testing API Integration

Before running the full video processor, you can test API integration with a mockup test:

```bash
python test_api_mockup.py
```

This will:
- Fetch a task from the API
- Simulate all processing steps without actual video processing
- Test all API endpoints (status updates, route updates, completion, etc.)
- Help verify API connectivity and endpoint correctness

### Running the Full Video Processor

Run the video processor:

```bash
python video_processor.py
```

The script will:
- Poll the API every 15 seconds (configurable) for new tasks
- Process each task through the complete pipeline
- Report results back to the API
- Clean up local files after processing

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
- Environment variables (`MEDIA_MODEL_DIR`, `STELLA_EXECUTABLE`, `STELLA_CONFIG_PATH`, `STELLA_VOCAB_PATH`) must be set in `.env`

## Development

### TODO Items

Some functions need implementation:
- `stitch_videos()`: Implement actual video stitching logic
- `calculate_route()`: Implement GPS/route extraction from frames
- Frame quality analysis for best frame selection
- Cloud storage integration details

## License

[Add your license here]
