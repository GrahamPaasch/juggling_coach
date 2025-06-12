from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple

@dataclass
class ThrowMetrics:
    height: float
    dwell_time: float
    flight_time: float
    throw_angle: float

@dataclass
class AdvancedMetrics:
    throw_height_consistency: float  # σ of y-max
    horizontal_drift: float         # σ of x landing points
    beat_timing_error: float       # Δt between catches
    dwell_ratio: float            # dwell time / cycle time
    pattern_symmetry: float       # left/right path difference

class PatternAnalyzer:
    def __init__(self, fps: float = 30.0, gravity: float = 9.81):
        self.fps = fps
        self.gravity = gravity
        
    def calculate_throw_metrics(self, positions: List[Tuple[int, int]]) -> ThrowMetrics:
        """Calculate metrics for a single throw."""
        if len(positions) < 3:
            return None
            
        # Convert positions to numpy array for easier calculation
        positions = np.array(positions)
        
        # Calculate height (maximum y displacement)
        height = abs(max(positions[:, 1]) - min(positions[:, 1])) / 100  # Convert to meters
        
        # Calculate flight time
        flight_time = len(positions) / self.fps
        
        # Estimate dwell time (time ball spends in hand)
        # Using velocity changes to detect catches/throws
        velocities = np.diff(positions, axis=0)
        dwell_time = self._estimate_dwell_time(velocities) / self.fps
        
        # Calculate throw angle
        throw_angle = self._calculate_throw_angle(positions[:3])
        
        return ThrowMetrics(
            height=height,
            dwell_time=dwell_time,
            flight_time=flight_time,
            throw_angle=throw_angle
        )
    
    def _estimate_dwell_time(self, velocities: np.ndarray) -> int:
        """Estimate frames where ball is being held (low velocity)."""
        speeds = np.linalg.norm(velocities, axis=1)
        threshold = np.mean(speeds) * 0.3  # 30% of mean speed
        return np.sum(speeds < threshold)
    
    def _calculate_throw_angle(self, initial_positions: np.ndarray) -> float:
        """Calculate the initial throw angle in degrees."""
        if len(initial_positions) < 3:
            return 0.0
        
        dx = initial_positions[2][0] - initial_positions[0][0]
        dy = initial_positions[0][1] - initial_positions[2][1]  # y is inverted in image coordinates
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

class EnhancedPatternAnalyzer:
    def __init__(self, fps: float = 30.0, gravity: float = 9.81):
        self.fps = fps
        self.gravity = gravity
        self.session_metrics: List[AdvancedMetrics] = []
        self.ewma_alpha = 0.2  # Exponential weighted moving average factor

    def calculate_advanced_metrics(self, tracked_objects: Dict[int, List[Tuple[int, int]]]) -> AdvancedMetrics:
        """Calculate comprehensive metrics for juggling feedback."""
        # Calculate throw height consistency
        y_maxes = [max(obj_positions, key=lambda p: p[1])[1] 
                  for obj_positions in tracked_objects.values()]
        height_consistency = np.std(y_maxes)

        # Calculate horizontal drift
        landing_points = self._get_landing_points(tracked_objects)
        horizontal_drift = np.std([p[0] for p in landing_points])

        # Calculate beat timing
        catch_times = self._detect_catches(tracked_objects)
        timing_errors = np.diff(catch_times)
        beat_error = np.std(timing_errors - np.median(timing_errors))

        # Calculate dwell ratio
        dwell_frames = self._calculate_dwell_frames(tracked_objects)
        total_frames = max(len(pos) for pos in tracked_objects.values())
        dwell_ratio = dwell_frames / total_frames

        # Calculate pattern symmetry
        left_area, right_area = self._calculate_path_areas(tracked_objects)
        symmetry = abs(left_area - right_area) / (left_area + right_area)

        return AdvancedMetrics(
            throw_height_consistency=height_consistency,
            horizontal_drift=horizontal_drift,
            beat_timing_error=beat_error,
            dwell_ratio=dwell_ratio,
            pattern_symmetry=symmetry
        )

    def _get_landing_points(self, tracked_objects):
        """Extract landing points from trajectories."""
        landing_points = []
        for positions in tracked_objects.values():
            # Find local minima in y-coordinates
            y_coords = [p[1] for p in positions]
            minima_indices = self._find_local_minima(y_coords)
            landing_points.extend([positions[i] for i in minima_indices])
        return landing_points