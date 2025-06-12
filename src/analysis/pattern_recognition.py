from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from ..tracking.tracker import TrackedObject

class JugglingPattern(Enum):
    UNKNOWN = "Unknown"
    CASCADE = "Cascade"
    SHOWER = "Shower"
    COLUMNS = "Columns"
    FOUNTAIN = "Fountain"

@dataclass
class PatternInfo:
    pattern_type: JugglingPattern
    confidence: float
    num_balls: int

class PatternRecognizer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
    
    def recognize_pattern(self, tracked_objects: Dict[int, TrackedObject]) -> Optional[PatternInfo]:
        if len(tracked_objects) < 2:
            return None
            
        # Get recent positions for all objects
        positions_by_object = {
            obj_id: np.array(obj.positions[-self.window_size:])
            for obj_id, obj in tracked_objects.items()
            if len(obj.positions) >= self.window_size
        }
        
        if not positions_by_object:
            return None
            
        # Calculate pattern features
        horizontality = self._calculate_horizontality(positions_by_object)
        symmetry = self._calculate_symmetry(positions_by_object)
        crossing_ratio = self._calculate_crossing_ratio(positions_by_object)
        
        # Pattern classification
        if horizontality > 0.8 and symmetry < 0.3:
            return PatternInfo(JugglingPattern.SHOWER, 0.8, len(tracked_objects))
        elif crossing_ratio > 0.6 and symmetry > 0.7:
            return PatternInfo(JugglingPattern.CASCADE, 0.9, len(tracked_objects))
        elif horizontality < 0.3 and symmetry > 0.8:
            return PatternInfo(JugglingPattern.COLUMNS, 0.85, len(tracked_objects))
        elif horizontality > 0.6 and symmetry > 0.8:
            return PatternInfo(JugglingPattern.FOUNTAIN, 0.7, len(tracked_objects))
            
        return PatternInfo(JugglingPattern.UNKNOWN, 0.5, len(tracked_objects))
    
    def _calculate_horizontality(self, positions_by_object: Dict[int, np.ndarray]) -> float:
        """Measure how horizontal the pattern is (shower-like patterns)."""
        all_trajectories = np.concatenate(list(positions_by_object.values()))
        if len(all_trajectories) < 2:
            return 0.0
        x_spread = np.std(all_trajectories[:, 0])
        y_spread = np.std(all_trajectories[:, 1])
        return x_spread / (x_spread + y_spread) if (x_spread + y_spread) > 0 else 0.0
    
    def _calculate_symmetry(self, positions_by_object: Dict[int, np.ndarray]) -> float:
        """Measure pattern symmetry (cascade-like patterns)."""
        all_positions = np.concatenate(list(positions_by_object.values()))
        if len(all_positions) < 2:
            return 0.0
        center_x = np.mean(all_positions[:, 0])
        left_points = all_positions[all_positions[:, 0] < center_x]
        right_points = all_positions[all_positions[:, 0] >= center_x]
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.0
        return min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
    
    def _calculate_crossing_ratio(self, positions_by_object: Dict[int, np.ndarray]) -> float:
        """Measure how often trajectories cross (cascade vs columns)."""
        if len(positions_by_object) < 2:
            return 0.0
        crossings = 0
        total_comparisons = 0
        
        for id1, pos1 in positions_by_object.items():
            for id2, pos2 in positions_by_object.items():
                if id1 >= id2:
                    continue
                crossings += self._count_trajectory_crossings(pos1, pos2)
                total_comparisons += 1
        
        return crossings / (total_comparisons * self.window_size) if total_comparisons > 0 else 0.0
    
    def _count_trajectory_crossings(self, traj1: np.ndarray, traj2: np.ndarray) -> int:
        """Count how many times two trajectories cross."""
        crossings = 0
        for i in range(len(traj1) - 1):
            for j in range(len(traj2) - 1):
                if self._segments_intersect(traj1[i], traj1[i+1], traj2[j], traj2[j+1]):
                    crossings += 1
        return crossings
    
    def _segments_intersect(self, p1: np.ndarray, p2: np.ndarray, 
                          p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)