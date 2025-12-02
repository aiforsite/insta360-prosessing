"""
Route calculation module for calculating routes using Stella VSLAM.
"""

import logging
import math
import subprocess
import json
import shutil
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RouteCalculation:
    """Handles route calculation using Stella VSLAM."""
    
    def __init__(
        self,
        work_dir: Path,
        stella_exec: str,
        stella_config_path: str,
        stella_vocab_path: str,
        stella_results_path: str,
        candidates_per_second: int,
        use_wsl: bool = False,
        use_docker: bool = False,
        docker_image: str = "stella_vslam-socket",
        docker_data_mount: str = "/data"
    ):
        """Initialize route calculation."""
        self.work_dir = work_dir
        self.stella_exec = stella_exec
        self.stella_config_path = stella_config_path
        self.stella_vocab_path = stella_vocab_path
        self.stella_results_path = stella_results_path
        self.candidates_per_second = candidates_per_second
        self.use_wsl = use_wsl
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_data_mount = docker_data_mount
    
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
    
    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Duration in seconds, or None if unable to determine
        """
        try:
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                duration = probe_data.get('format', {}).get('duration')
                if duration:
                    duration_float = float(duration)
                    logger.info(f"Video duration: {duration_float:.2f} seconds ({duration_float / 60:.2f} minutes)")
                    return duration_float
            logger.warning(f"Could not extract duration from ffprobe output")
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffprobe not available or timeout: {e}")
            return None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse ffprobe output: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error getting video duration: {e}")
            return None
    
    def _parse_route_lines(self, map_output_path: Path) -> List[str]:
        """Read raw route lines from map output file."""
        lines: List[str] = []
        try:
            with open(map_output_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                # Remove null characters from lines
                lines = [line.replace('\u0000', '').replace('\x00', '') for line in lines]
                return lines
        except UnicodeDecodeError:
            try:
                with open(map_output_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    # Remove null characters from lines
                    lines = [line.replace('\u0000', '').replace('\x00', '') for line in lines]
            except Exception as exc:
                logger.warning(f"Failed to parse map file {map_output_path}: {exc}")
        except Exception as exc:
            logger.warning(f"Failed to read map file {map_output_path}: {exc}")
        return lines
    
    def _build_line_mapping(self, route_lines: List[str]) -> Dict[int, str]:
        """Create mapping from line index to route line content."""
        mapping: Dict[int, str] = {}
        for line in route_lines:
            if not line or not line.strip():
                continue
            # Split line and get first token (should be frame index)
            parts = line.split()
            if not parts:
                continue
            token = parts[0]
            try:
                idx = int(float(token))
                # Only map if line has more than just the index (should have pose data)
                if len(parts) > 1:
                    mapping[idx] = line
                else:
                    logger.debug(f"Skipping line with only index: {line}")
            except ValueError:
                logger.debug(f"Could not parse index from line: {line[:50]}...")
                continue
        logger.info(f"Built line mapping with {len(mapping)} valid entries from {len(route_lines)} total lines")
        return mapping
    
    def _parse_stella_trajectory(self, stella_data: Dict) -> List[Tuple[float, str]]:
        """
        Parse frame_trajectory data to get all timestamp entries.
        
        Returns list of (timestamp, line) tuples sorted by timestamp.
        """
        frame_trajectory_str = stella_data.get("frame_trajectory", "")
        if not frame_trajectory_str:
            logger.warning("No frame_trajectory found in stella_data")
            return []
        
        frame_trajectory = [e for e in frame_trajectory_str.split("\n") if e]
        trajectory_entries = []
        
        logger.info(f"Processing {len(frame_trajectory)} frame_trajectory lines")
        for idx, line in enumerate(frame_trajectory):
            line = line.strip()
            if not line:
                continue
            
            try:
                frame_time = float(line.split()[0])
                trajectory_entries.append((frame_time, line))
                
                # Log first few lines to understand the data
                if idx < 10:
                    logger.debug(f"Trajectory line {idx}: frame_time={frame_time:.3f}s, line={line[:60]}...")
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse frame trajectory line {idx}: {line[:50]}... Error: {e}")
                continue
        
        # Sort by timestamp
        trajectory_entries.sort(key=lambda x: x[0])
        logger.info(f"Parsed {len(trajectory_entries)} trajectory entries")
        return trajectory_entries
    
    def _filter_frames_data_from_stella_data(self, stella_data: Dict) -> Tuple[Dict[int, Tuple[float, str]], str]:
        """
        Filter frame_trajectory data to get one entry per second.
        
        "frame_trajectory" contains more data points than one per second.
        Here we extract them for every second.
        
        Returns:
            Tuple of (filtered_data dict, formatted string)
        """
        trajectory_entries = self._parse_stella_trajectory(stella_data)
        if not trajectory_entries:
            return {}, ""
        
        filtered_data = {}
        
        # We make sure the first line is always 0, because sometimes frame_trajectory begins from 1.
        first_line = "0.0 -0 -0 -0 0 0 0 1"
        filtered_data[0] = (0.0, first_line)
        
        for frame_time, line in trajectory_entries:
            # We use the floor() function to ensure that we will never round up to
            # a frame index that will not exist.
            time_in_video = int(math.floor(frame_time))
            
            # We take the first frame with a timestamp beginning with a new second.
            if time_in_video not in filtered_data:
                filtered_data[time_in_video] = (frame_time, line)
        
        # Build formatted string
        s = ""
        for _, (_, line) in sorted(filtered_data.items()):
            s += line
            s += "\n"
        
        logger.info(f"Filtered {len(filtered_data)} entries from {len(trajectory_entries)} trajectory entries")
        logger.debug(f"Filtered data keys: {sorted(filtered_data.keys())}")
        return filtered_data, s.strip()
    
    def _print_frame_trajectory_analysis(
        self, 
        stella_data: Dict, 
        filtered_data: Dict[int, Tuple[float, str]], 
        raw_path: Dict[str, List],
        frame_count: int
    ):
        """Print analysis of frame trajectory data before saving route."""
        print("\n" + "="*80)
        print("FRAME TRAJECTORY ANALYSIS")
        print("="*80)
        
        # Basic statistics
        frame_trajectory_str = stella_data.get("frame_trajectory", "")
        if frame_trajectory_str:
            frame_trajectory_lines = [e for e in frame_trajectory_str.split("\n") if e]
            print(f"Total frame_trajectory entries: {len(frame_trajectory_lines)}")
        else:
            print("No frame_trajectory data found in stella_data")
        
        print(f"Filtered entries (one per second): {len(filtered_data)}")
        print(f"Total frames processed: {frame_count}")
        
        # Raw path statistics
        valid_entries = sum(1 for entry in raw_path.values() if entry[1] and entry[1].strip())
        print(f"Raw path entries with valid route data: {valid_entries}/{len(raw_path)}")
        
        # Time range analysis
        if filtered_data:
            times = sorted([t for t, _ in filtered_data.values()])
            if times:
                print(f"Time range: {times[0]:.2f}s - {times[-1]:.2f}s (duration: {times[-1] - times[0]:.2f}s)")
        
        # Sample trajectory data
        print("\nSample trajectory entries (first 5):")
        for i, (time_idx, (frame_time, line)) in enumerate(sorted(filtered_data.items())[:5]):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    print(f"  [{time_idx}s] t={frame_time:.3f}s: pos=({x:.3f}, {y:.3f}, {z:.3f})")
                except (ValueError, IndexError):
                    print(f"  [{time_idx}s] t={frame_time:.3f}s: {line[:60]}...")
            else:
                print(f"  [{time_idx}s] t={frame_time:.3f}s: {line[:60]}...")
        
        # Coverage analysis
        if raw_path:
            coverage_percent = (valid_entries / len(raw_path)) * 100 if raw_path else 0
            print(f"\nRoute coverage: {coverage_percent:.1f}% ({valid_entries}/{len(raw_path)} seconds have route data)")
        
        print("="*80 + "\n")
        logger.info(f"Frame trajectory analysis: {len(filtered_data)} filtered entries, {valid_entries}/{len(raw_path)} seconds with route data")
    
    def _is_valid_route_line(self, line: str) -> bool:
        """Check if a route line is valid (has pose data, not just index)."""
        if not line or not line.strip():
            return False
        parts = line.split()
        # Valid line should have at least 2 parts (index + pose data)
        # Pose data typically has 7-8 values (position + quaternion)
        if len(parts) < 2:
            return False
        # Check that line doesn't contain only control characters or garbage
        # Remove nulls and check if there's actual content
        cleaned = line.replace('\u0000', '').replace('\x00', '').strip()
        if len(cleaned) < 10:  # Minimum reasonable length for pose data
            return False
        return True
    
    def _build_raw_path(self, video_path: Path, video_duration: float, trajectory_entries: List[Tuple[float, str]]) -> Dict[str, List]:
        """
        Produce raw_path dict with one coordinate per second.
        
        For each second in the video (0, 1, 2, ..., duration), finds the closest
        timestamp in Stella trajectory data and uses that coordinate.
        
        Args:
            video_path: Path to video file (for logging)
            video_duration: Video duration in seconds
            trajectory_entries: List of (timestamp, route_line) tuples from Stella
        
        Returns:
            Dict with structure: {"0": [0.0, "coords"], "1": [1.0, "coords"], ...}
        """
        raw_path: Dict[str, List] = {}
        
        if not trajectory_entries:
            logger.warning("No trajectory entries available for raw_path")
            return raw_path
        
        # Calculate number of seconds (rounded up)
        num_seconds = int(math.ceil(video_duration))
        logger.info(f"Building raw_path for {num_seconds} seconds (video duration: {video_duration:.2f}s) using {len(trajectory_entries)} trajectory entries")
        
        # Create mapping from timestamp to line for quick lookup
        timestamp_to_line = {ts: line for ts, line in trajectory_entries}
        timestamps = sorted(timestamp_to_line.keys())
        
        if not timestamps:
            logger.warning("No valid timestamps found in trajectory data")
            return raw_path
        
        # For each second in the video, find the closest timestamp
        for second in range(num_seconds):
            target_time = float(second)
            
            # Find closest timestamp
            closest_timestamp = min(timestamps, key=lambda ts: abs(ts - target_time))
            closest_line = timestamp_to_line[closest_timestamp]
            
            # Remove null characters (\u0000) that cause PostgreSQL text field errors
            cleaned_line = closest_line.replace('\u0000', '').replace('\x00', '')
            
            raw_path[str(second)] = [target_time, cleaned_line]
            
            # Debug logging for first 10 entries
            if second < 10:
                logger.debug(f"Second {second}: time={target_time:.1f}s, closest_timestamp={closest_timestamp:.3f}s, diff={abs(closest_timestamp - target_time):.3f}s")
        
        # Log statistics
        valid_count = sum(1 for entry in raw_path.values() if entry[1] and entry[1].strip())
        logger.info(f"Built raw_path with {valid_count}/{len(raw_path)} entries having valid route data")
        
        return raw_path
    
    def calculate_route(self, frame_paths: List[Path], video_path: Path, update_status_callback) -> Optional[Dict]:
        """Calculate route using selected frames (Stella VSLAM) and update raw path.
        
        Args:
            frame_paths: List of frame paths for Stella processing
            video_path: Path to stitched video file (for duration and raw_path generation)
            update_status_callback: Status update callback
        
        Returns:
            Dict with route data including raw_path and duration, or None if failed
        """
        logger.info("Calculating route from frames using Stella VSLAM...")
        update_status_callback(f"Calculating route using selected frames ({len(frame_paths)} frames)...")
        
        if not frame_paths:
            logger.warning("No frames available for route calculation")
            update_status_callback("Error in route calculation: frames not available")
            return None
        
        # Get video duration
        video_duration = self._get_video_duration(video_path)
        if video_duration is None:
            logger.warning("Could not determine video duration, will use trajectory data range")
            # Fallback: will be determined from trajectory data
        
        if not all([self.stella_exec, self.stella_config_path, self.stella_vocab_path]):
            error_msg = "Stella VSLAM environment variables (STELLA_EXECUTABLE, STELLA_CONFIG_PATH, STELLA_VOCAB_PATH) must be set"
            logger.error(error_msg)
            update_status_callback(f"Error in route calculation: {error_msg}")
            return None
        
        frames_dir = frame_paths[0].parent
        map_output_path = self.work_dir / "stella_map.msg"
        if map_output_path.exists():
            map_output_path.unlink()
        
        # Ensure stella_results_path directory exists
        if self.stella_results_path:
            stella_results_dir = Path(self.stella_results_path)
            # Handle relative paths (e.g., ./work/stella_results)
            if not stella_results_dir.is_absolute():
                stella_results_dir = self.work_dir / stella_results_dir
            stella_results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured stella_results directory exists: {stella_results_dir}")
            # Use absolute path for Stella command
            stella_results_path_abs = str(stella_results_dir.resolve())
        else:
            stella_results_path_abs = ""
        
        # Convert Windows paths to WSL paths for Docker volume mounts
        def to_wsl_path(path: str) -> str:
            # If already a Linux/WSL path (starts with /), return as-is
            if path.startswith('/'):
                return path
            # Windows path like C:\work\frames or C:/work/frames
            if ':' in path:
                drive = path[0].lower()
                rest = path[2:].replace('\\', '/')
                # Remove leading slash if present
                if rest.startswith('/'):
                    rest = rest[1:]
                return f"/mnt/{drive}/{rest}"
            # Relative path, just normalize separators
            return path.replace('\\', '/')
        
        # Verify WSL and Docker are available when using Docker
        def verify_wsl_docker():
            """Verify that WSL and Docker are available."""
            try:
                # Check if WSL is available - try multiple methods for Task Scheduler compatibility
                wsl_available = False
                
                # Method 1: Try wsl --list --quiet (may fail in Task Scheduler)
                try:
                    wsl_check = subprocess.run(
                        ['wsl', '--list', '--quiet'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    # Accept any return code - wsl --list may return non-zero but WSL is still available
                    wsl_available = True
                    logger.debug("WSL check via --list succeeded")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                # Method 2: Try simple wsl command (more reliable in Task Scheduler)
                if not wsl_available:
                    try:
                        wsl_check2 = subprocess.run(
                            ['wsl', 'echo', 'test'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if wsl_check2.returncode == 0:
                            wsl_available = True
                            logger.debug("WSL check via echo succeeded")
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                
                if not wsl_available:
                    raise Exception("WSL command not found or not accessible. Make sure WSL is installed and in system PATH.")
                
                # Check if Docker is available in WSL
                # Use --exec to ensure we're in WSL context
                docker_check = subprocess.run(
                    ['wsl', '--exec', 'docker', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if docker_check.returncode != 0:
                    # Try alternative: wsl docker --version (without --exec)
                    docker_check2 = subprocess.run(
                        ['wsl', 'docker', '--version'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if docker_check2.returncode != 0:
                        raise Exception("Docker is not available in WSL. Make sure Docker is installed in WSL.")
                    else:
                        logger.debug("Docker check via 'wsl docker' succeeded")
                else:
                    logger.debug("Docker check via 'wsl --exec docker' succeeded")
                
                logger.info("WSL and Docker verified successfully")
                return True
            except FileNotFoundError:
                raise Exception("WSL command not found in PATH. Make sure WSL is installed and in system PATH.")
            except subprocess.TimeoutExpired:
                raise Exception("WSL or Docker check timed out. This may indicate WSL is not running or Docker is not accessible.")
            except Exception as e:
                raise Exception(f"WSL/Docker verification failed: {e}")
        
        # Build command based on execution method
        if self.use_docker:
            # Verify WSL and Docker are available
            try:
                verify_wsl_docker()
            except Exception as e:
                error_msg = f"WSL/Docker verification failed: {e}"
                logger.error(error_msg)
                # Clean error message for status update
                cleaned_error = str(e).replace('\u0000', '').replace('\x00', '').replace('\r', ' ').replace('\n', ' ').strip()
                update_status_callback(f"Error: {cleaned_error}")
                return None
            
            # Docker execution via WSL: mount work_dir to /data in container
            # Use absolute path and ensure it exists
            work_dir_abs = str(self.work_dir.resolve())
            if not self.work_dir.exists():
                error_msg = f"Work directory does not exist: {work_dir_abs}"
                logger.error(error_msg)
                update_status_callback(f"Error: {error_msg}")
                return None
            work_dir_wsl = to_wsl_path(work_dir_abs)
            
            # Paths inside container
            # Config and vocab are already in the container at /opt/stella_vslam/
            # Use them directly, no need to copy to /data
            frames_dir_container = f"{self.docker_data_mount}/{frames_dir.relative_to(self.work_dir)}"
            map_output_path_container = f"{self.docker_data_mount}/{map_output_path.relative_to(self.work_dir)}"
            stella_results_path_container = None
            if stella_results_path_abs:
                stella_results_path_container = f"{self.docker_data_mount}/{Path(stella_results_path_abs).relative_to(self.work_dir)}"
            
            # Use container-internal paths for config and vocab (they're already in the image)
            # config.json has paths like /opt/stella_vslam/insta360_equirect.yaml
            stella_config_path_container = self.stella_config_path
            stella_vocab_path_container = self.stella_vocab_path
            
            # Build Docker command via WSL: wsl docker run ...
            # Use --entrypoint to override Dockerfile entrypoint
            # WSL paths are already in Linux format, so use them directly
            cmd = [
                'wsl', 'docker', 'run', '--rm',
                '-v', f'{work_dir_wsl}:{self.docker_data_mount}',
                '-v', '/opt/stella_vslam:/opt/stella_vslam:ro',
                '--workdir', '/stella_vslam_examples/build',
                '--entrypoint', self.stella_exec,
                self.docker_image,
                '-c', stella_config_path_container,
                '-d', frames_dir_container,
                '--frame-skip', '1',
                '--no-sleep',
                '--start-timestamp', '0',
                '--auto-term',
                '--map-db-out', map_output_path_container,
                '-v', stella_vocab_path_container
            ]
            
            if stella_results_path_container:
                cmd.extend(['--eval-log-dir', stella_results_path_container])
                
        elif self.use_wsl:
            # WSL execution
            # Convert work_dir paths (Windows) to WSL paths
            frames_dir_wsl = to_wsl_path(str(frames_dir))
            map_output_path_wsl = to_wsl_path(str(map_output_path))
            stella_results_path_wsl = to_wsl_path(stella_results_path_abs) if stella_results_path_abs else None
            
            # Stella config and vocab paths are already Linux paths in config, use as-is
            stella_config_path_wsl = self.stella_config_path
            stella_vocab_path_wsl = self.stella_vocab_path
            
            # Build WSL command: wsl <stella_exec> <args>
            cmd = [
                'wsl',
                self.stella_exec,
                '-c', stella_config_path_wsl,
                '-d', frames_dir_wsl,
                '--frame-skip', '1',
                '--no-sleep',
                '--start-timestamp', '0',
                '--auto-term',
                '--map-db-out', map_output_path_wsl,
                '-v', stella_vocab_path_wsl
            ]
            
            if stella_results_path_wsl:
                cmd.extend(['--eval-log-dir', stella_results_path_wsl])
        else:
            # Native execution (Linux)
            cmd = [
                self.stella_exec,
                '-c', self.stella_config_path,
                '-d', str(frames_dir),
                '--frame-skip', '1',
                '--no-sleep',
                '--start-timestamp', '0',
                '--auto-term',
                '--map-db-out', str(map_output_path),
                '-v', self.stella_vocab_path
            ]
            
            # Add --eval-log-dir only if path is set
            if stella_results_path_abs:
                cmd.extend(['--eval-log-dir', stella_results_path_abs])
        
        logger.info(f"Running Stella VSLAM command: {' '.join(cmd)}")
        try:
            # Set working directory to ensure relative paths work correctly
            # Use absolute path for work_dir
            # For Task Scheduler compatibility, ensure PATH includes system directories
            env = os.environ.copy()
            # Add common WSL/Docker paths if not already in PATH
            system_paths = [
                r'C:\Windows\System32',
                r'C:\Windows',
                r'C:\Windows\System32\WindowsPowerShell\v1.0'
            ]
            current_path = env.get('PATH', '')
            for path in system_paths:
                if path not in current_path:
                    current_path = f"{path};{current_path}"
            env['PATH'] = current_path
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(self.work_dir.resolve()) if self.work_dir.exists() else None,
                env=env
            )
            
            if result.stdout:
                logger.info(f"Stella VSLAM output: {result.stdout[:500]}...")  # Log first 500 chars
            if result.stderr:
                logger.warning(f"Stella VSLAM stderr: {result.stderr}")
            
            if not map_output_path.exists():
                raise FileNotFoundError(f"Expected map output not found at {map_output_path}")
            
            # Try to get frame_trajectory data from various sources
            stella_data = {}
            
            # Try 1: Parse JSON from stdout
            try:
                if result.stdout:
                    stdout_json = json.loads(result.stdout)
                    if isinstance(stdout_json, dict) and "frame_trajectory" in stdout_json:
                        stella_data = stdout_json
                        logger.info("Found frame_trajectory in stdout JSON")
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Try 2: Look for JSON file in work directory
            if not stella_data.get("frame_trajectory"):
                json_files = list(self.work_dir.glob("*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            if isinstance(json_data, dict) and "frame_trajectory" in json_data:
                                stella_data = json_data
                                logger.info(f"Found frame_trajectory in {json_file}")
                                break
                    except (json.JSONDecodeError, ValueError, IOError):
                        continue
            
            # Try 3: Read from frame_trajectory.txt in STELLA_RESULTS_PATH
            if not stella_data.get("frame_trajectory") and self.stella_results_path:
                # Handle relative paths (e.g., ./work/stella_results)
                stella_results_dir = Path(self.stella_results_path)
                if not stella_results_dir.is_absolute():
                    stella_results_dir = self.work_dir / stella_results_dir
                
                frame_trajectory_path = stella_results_dir / "frame_trajectory.txt"
                keyframe_trajectory_path = stella_results_dir / "keyframe_trajectory.txt"
                
                # Try frame_trajectory.txt first
                if frame_trajectory_path.exists():
                    try:
                        with open(frame_trajectory_path, 'r', encoding='utf-8') as f:
                            frame_trajectory_content = f.read()
                            stella_data = {"frame_trajectory": frame_trajectory_content}
                            logger.info(f"Found frame_trajectory in {frame_trajectory_path}")
                    except (IOError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to read frame_trajectory.txt: {e}")
                # Fallback to keyframe_trajectory.txt
                elif keyframe_trajectory_path.exists():
                    try:
                        with open(keyframe_trajectory_path, 'r', encoding='utf-8') as f:
                            frame_trajectory_content = f.read()
                            stella_data = {"frame_trajectory": frame_trajectory_content}
                            logger.info(f"Found keyframe_trajectory in {keyframe_trajectory_path}")
                    except (IOError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to read keyframe_trajectory.txt: {e}")
            
            # Parse trajectory entries from stella_data
            trajectory_entries = []
            if stella_data.get("frame_trajectory"):
                trajectory_entries = self._parse_stella_trajectory(stella_data)
            else:
                # Try 4: Parse from map.msg file (fallback to old method)
                logger.warning("No frame_trajectory found in JSON, falling back to map.msg parsing")
                route_lines = self._parse_route_lines(map_output_path)
                line_mapping = self._build_line_mapping(route_lines)
                for idx, line in line_mapping.items():
                    try:
                        frame_time = float(line.split()[0])
                        trajectory_entries.append((frame_time, line))
                    except (ValueError, IndexError):
                        continue
                # Ensure first entry is 0
                if not trajectory_entries or trajectory_entries[0][0] > 0.1:
                    trajectory_entries.insert(0, (0.0, "0.0 -0 -0 -0 0 0 0 1"))
                trajectory_entries.sort(key=lambda x: x[0])
            
            # If video duration not available, use max timestamp from trajectory
            if video_duration is None and trajectory_entries:
                max_timestamp = max(ts for ts, _ in trajectory_entries)
                video_duration = max_timestamp
                logger.info(f"Using trajectory max timestamp as video duration: {video_duration:.2f}s")
            
            # Build raw_path using video duration and trajectory entries
            if video_duration and trajectory_entries:
                raw_path = self._build_raw_path(video_path, video_duration, trajectory_entries)
            else:
                logger.warning("Cannot build raw_path: missing video duration or trajectory data")
                raw_path = {}
            
            # Get filtered_data for analysis (backward compatibility)
            filtered_data, _ = self._filter_frames_data_from_stella_data(stella_data) if stella_data.get("frame_trajectory") else ({}, "")
            
            # Print frame trajectory analysis before saving
            self._print_frame_trajectory_analysis(stella_data, filtered_data, raw_path, len(frame_paths))
            
            route_data = {
                'map_file': str(map_output_path),
                'frame_count': len(frame_paths),
                'generated_at': datetime.now().isoformat(),
                'command': cmd,
                'raw_path': raw_path,
                'duration': video_duration
            }
            
            logger.info("Route calculated")
            update_status_callback("Reitti laskettu, päivitetään raw path tulosten avulla...")
            return route_data
        except subprocess.CalledProcessError as e:
            logger.error(f"Stella VSLAM failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            
            # Clean error message: remove null characters and other control characters
            error_msg = e.stderr or str(e) if e else "Unknown error"
            cleaned_error = error_msg.replace('\u0000', '').replace('\x00', '').replace('\r', ' ').replace('\n', ' ').strip()
            if len(cleaned_error) > 200:
                cleaned_error = cleaned_error[:197] + "..."
            update_status_callback(f"Error in route calculation: {cleaned_error}")
            return None
        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            update_status_callback(f"Error in route calculation: {str(e)}")
            return None

