import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ThrowMetrics:
    height: float
    dwell_time: float
    flight_time: float
    throw_angle: float

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