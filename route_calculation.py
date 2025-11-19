"""
Route calculation module for calculating routes using Stella VSLAM.
"""

import logging
import math
import subprocess
import json
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
        candidates_per_second: int
    ):
        """Initialize route calculation."""
        self.work_dir = work_dir
        self.stella_exec = stella_exec
        self.stella_config_path = stella_config_path
        self.stella_vocab_path = stella_vocab_path
        self.stella_results_path = stella_results_path
        self.candidates_per_second = candidates_per_second
    
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
    
    def _filter_frames_data_from_stella_data(self, stella_data: Dict) -> Tuple[Dict[int, Tuple[float, str]], str]:
        """
        Filter frame_trajectory data to get one entry per second.
        
        "frame_trajectory" contains more data points than one per second.
        Here we extract them for every second.
        
        Returns:
            Tuple of (filtered_data dict, formatted string)
        """
        frame_trajectory_str = stella_data.get("frame_trajectory", "")
        if not frame_trajectory_str:
            logger.warning("No frame_trajectory found in stella_data")
            return {}, ""
        
        frame_trajectory = [e for e in frame_trajectory_str.split("\n") if e]
        filtered_data = {}
        
        # We make sure the first line is always 0, because sometimes frame_trajectory begins from 1.
        first_line = "0.0 -0 -0 -0 0 0 0 1"
        filtered_data[0] = (0.0, first_line)
        
        for line in frame_trajectory:
            line = line.strip()
            if not line:
                continue
            
            try:
                frame_time = float(line.split()[0])
                # We use the floor() function to ensure that we will never round up to
                # a frame index that will not exist.
                time_in_video = math.floor(frame_time)
                
                # We take the first frame with a timestamp beginning with a new second.
                if time_in_video not in filtered_data:
                    # Append the 1st frame starting with a new second.
                    filtered_data[time_in_video] = (frame_time, line)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse frame trajectory line: {line[:50]}... Error: {e}")
                continue
        
        # Build formatted string
        s = ""
        for _, (_, line) in sorted(filtered_data.items()):
            s += line
            s += "\n"
        
        logger.info(f"Filtered {len(filtered_data)} entries from {len(frame_trajectory)} frame_trajectory lines")
        return filtered_data, s.strip()
    
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
    
    def _build_raw_path(self, frame_paths: List[Path], filtered_data: Dict[int, Tuple[float, str]]) -> Dict[str, List]:
        """
        Produce raw_path dict aligning filtered route data with selected frames.
        
        Uses filtered frame_trajectory data (one entry per second) and matches it
        to frames based on their time_in_video.
        """
        raw_path: Dict[str, List] = {}
        
        logger.info(f"Building raw_path for {len(frame_paths)} frames using {len(filtered_data)} filtered route entries")
        
        for idx, frame_path in enumerate(frame_paths):
            suffix = self._extract_suffix(frame_path)
            time_in_video = (
                suffix / float(self.candidates_per_second)
                if suffix is not None and self.candidates_per_second
                else float(idx)
            )
            
            # Use floor() to match the filtering logic - ensures we never round up
            # to a frame index that will not exist
            time_index = math.floor(time_in_video)
            
            # Get the route line for this time index
            if time_index in filtered_data:
                frame_time, line = filtered_data[time_index]
                # Remove null characters (\u0000) that cause PostgreSQL text field errors
                line = line.replace('\u0000', '').replace('\x00', '')
            else:
                # If no route data for this second, use empty string
                logger.debug(f"Frame {idx} (time {time_in_video:.3f}s, index {time_index}): no route data found")
                line = ""
            
            raw_path[str(idx)] = [time_in_video, line]
        
        # Log statistics
        valid_count = sum(1 for entry in raw_path.values() if entry[1] and entry[1].strip())
        logger.info(f"Built raw_path with {valid_count}/{len(raw_path)} entries having valid route data")
        
        return raw_path
    
    def calculate_route(self, frame_paths: List[Path], update_status_callback) -> Optional[Dict]:
        """Calculate route using selected frames (Stella VSLAM) and update raw path."""
        logger.info("Calculating route from frames using Stella VSLAM...")
        update_status_callback(f"Lasketaan reitti valittujen frameiden avulla ({len(frame_paths)} framea)...")
        
        if not frame_paths:
            logger.warning("No frames available for route calculation")
            update_status_callback("Virhe reitin laskennassa: frameja ei saatavilla")
            return None
        
        if not all([self.stella_exec, self.stella_config_path, self.stella_vocab_path]):
            error_msg = "Stella VSLAM environment variables (STELLA_EXECUTABLE, STELLA_CONFIG_PATH, STELLA_VOCAB_PATH) must be set"
            logger.error(error_msg)
            update_status_callback(f"Virhe reitin laskennassa: {error_msg}")
            return None
        
        frames_dir = frame_paths[0].parent
        map_output_path = self.work_dir / "stella_map.msg"
        if map_output_path.exists():
            map_output_path.unlink()
        
        cmd = [
            self.stella_exec,
            '-c', self.stella_config_path,
            '-d', str(frames_dir),
            '--frame-skip', '1',
            '--no-sleep',
            '--auto-term',
            '--map-db-out', str(map_output_path),
            '--eval-log-dir', self.stella_results_path,
            '-v', self.stella_vocab_path
        ]
        
        logger.info(f"Running Stella VSLAM command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
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
                frame_trajectory_path = Path(self.stella_results_path) / "frame_trajectory.txt"
                keyframe_trajectory_path = Path(self.stella_results_path) / "keyframe_trajectory.txt"
                
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
            
            # Try 4: Parse from map.msg file (fallback to old method)
            if not stella_data.get("frame_trajectory"):
                logger.warning("No frame_trajectory found in JSON, falling back to map.msg parsing")
                route_lines = self._parse_route_lines(map_output_path)
                line_mapping = self._build_line_mapping(route_lines)
                # Convert to filtered_data format for compatibility
                filtered_data = {}
                for idx, line in line_mapping.items():
                    try:
                        frame_time = float(line.split()[0])
                        filtered_data[int(math.floor(frame_time))] = (frame_time, line)
                    except (ValueError, IndexError):
                        continue
                # Ensure first entry is 0
                if 0 not in filtered_data:
                    filtered_data[0] = (0.0, "0.0 -0 -0 -0 0 0 0 1")
            else:
                # Use frame_trajectory data with filtering
                filtered_data, _ = self._filter_frames_data_from_stella_data(stella_data)
            
            # Build raw_path using filtered data
            raw_path = self._build_raw_path(frame_paths, filtered_data) if filtered_data else {}
            
            route_data = {
                'map_file': str(map_output_path),
                'frame_count': len(frame_paths),
                'generated_at': datetime.now().isoformat(),
                'command': cmd,
                'raw_path': raw_path
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
            update_status_callback(f"Virhe reitin laskennassa: {e.stderr or str(e)}")
            return None
        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            update_status_callback(f"Virhe reitin laskennassa: {str(e)}")
            return None

