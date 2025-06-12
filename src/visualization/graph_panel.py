import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from ..training.statistics import StatsTracker

@dataclass
class GraphState:
    visible_metrics: Dict[str, bool] = field(default_factory=lambda: {
        'height_consistency': True,
        'horizontal_drift': True,
        'beat_timing': True,
        'dwell_ratio': True,
        'pattern_symmetry': True
    })
    time_window: int = 100
    selected_metric: Optional[str] = None
    is_dragging: bool = False
    last_mouse_pos: Tuple[int, int] = (0, 0)
    zoom_level: float = 1.0
    pan_offset: int = 0

class GraphPanel:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.stats = StatsTracker()
        self.state = GraphState()
        
    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        """Handle mouse events for graph interaction."""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.state.is_dragging = True
                self.state.last_mouse_pos = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.state.is_dragging = False
                
            elif event == cv2.EVENT_MOUSEMOVE and self.state.is_dragging:
                dx = x - self.state.last_mouse_pos[0]
                self._handle_pan(dx)
                self.state.last_mouse_pos = (x, y)
                
            elif event == cv2.EVENT_MOUSEWHEEL:
                self._handle_zoom(flags > 0)
        except Exception as e:
            print(f"Error in graph mouse handling: {e}")
    
    def _handle_pan(self, dx: int) -> None:
        """Handle panning of graph view."""
        # Implementation for panning
        pass
        
    def _handle_zoom(self, zoom_in: bool) -> None:
        """Handle zooming of graph view."""
        # Implementation for zooming
        pass
        
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render graph panel on frame."""
        # Implementation for rendering
        return frame