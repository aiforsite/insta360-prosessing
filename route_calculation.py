"""
Route calculation module for calculating routes using Stella VSLAM.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
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
        candidates_per_second: int
    ):
        """Initialize route calculation."""
        self.work_dir = work_dir
        self.stella_exec = stella_exec
        self.stella_config_path = stella_config_path
        self.stella_vocab_path = stella_vocab_path
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
                return lines
        except UnicodeDecodeError:
            try:
                with open(map_output_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
            except Exception as exc:
                logger.warning(f"Failed to parse map file {map_output_path}: {exc}")
        except Exception as exc:
            logger.warning(f"Failed to read map file {map_output_path}: {exc}")
        return lines
    
    def _build_line_mapping(self, route_lines: List[str]) -> Dict[int, str]:
        """Create mapping from line index to route line content."""
        mapping: Dict[int, str] = {}
        for line in route_lines:
            token = line.split()[0] if line else ""
            try:
                idx = int(float(token))
                mapping[idx] = line
            except ValueError:
                continue
        return mapping
    
    def _build_raw_path(self, frame_paths: List[Path], route_lines: List[str]) -> Dict[str, List]:
        """Produce raw_path dict aligning route lines with selected frames."""
        raw_path: Dict[str, List] = {}
        line_mapping = self._build_line_mapping(route_lines)
        
        for idx, frame_path in enumerate(frame_paths):
            suffix = self._extract_suffix(frame_path)
            time_in_video = (
                suffix / float(self.candidates_per_second)
                if suffix is not None and self.candidates_per_second
                else float(idx)
            )
            expected_idx = int(round(time_in_video))
            line = line_mapping.get(expected_idx)
            if line is None and idx < len(route_lines):
                line = route_lines[idx]
            if line is None:
                line = ""
            raw_path[str(idx)] = [time_in_video, line]
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
                logger.info(f"Stella VSLAM output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Stella VSLAM stderr: {result.stderr}")
            
            if not map_output_path.exists():
                raise FileNotFoundError(f"Expected map output not found at {map_output_path}")
            
            route_lines = self._parse_route_lines(map_output_path)
            raw_path = self._build_raw_path(frame_paths, route_lines) if route_lines else {}
            
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

