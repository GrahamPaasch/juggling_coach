import cv2
from math import cos, sin, radians
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from ..tracking.tracker import TrackedObject
from ..analysis.pattern_recognition import PatternInfo
from ..training.drill_generator import DrillExercise
from ..analysis.metrics import AdvancedMetrics
from ..training.statistics import StatsTracker

@dataclass
class View3DState:
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    zoom: float = 1.0
    is_dragging: bool = False
    last_mouse_pos: Tuple[int, int] = (0, 0)

@dataclass
class DrillVisualization:
    active_drill: DrillExercise = None
    start_time: float = 0
    progress: float = 0.0
    success_count: int = 0
    attempt_count: int = 0

@dataclass
class GraphState:
    visible_metrics: Dict[str, bool] = None
    time_window: int = 100  # Number of frames to show
    selected_metric: Optional[str] = None
    is_dragging: bool = False
    last_mouse_pos: Tuple[int, int] = (0, 0)
    zoom_level: float = 1.0
    pan_offset: int = 0

    def __post_init__(self):
        if self.visible_metrics is None:
            self.visible_metrics = {
                'height_consistency': True,
                'horizontal_drift': True,
                'beat_timing': True,
                'dwell_ratio': True,
                'pattern_symmetry': True
            }

class PatternVisualizer:
    def __init__(self):
        self.colors = {
            'trajectory': (0, 255, 0),
            'ball': (0, 255, 0),
            'text': (255, 255, 0),
            'pattern': (0, 255, 255),
            'metrics': (0, 255, 255),
            'heatmap': (0, 0, 255),
            'speed': (255, 165, 0)
        }
        self.colors.update({
            '3d_grid': (128, 128, 128),
            '3d_trajectory': (0, 255, 128)
        })
        self.heatmap = None
        self.stats_history = []
        self.perspective_matrix = None
        self.trajectory_3d_history = {}
        self.view_3d = View3DState()
        self.view_3d_region = (0, 0, 0, 0)  # x, y, width, height
        self.ideal_trajectory_cache = {}
        self.error_threshold = 20  # pixels
        self.drill_viz = DrillVisualization()
        self.drill_colors = {
            'progress_bar': (0, 255, 0),
            'instruction': (255, 255, 255),
            'success': (0, 255, 0),
            'failure': (0, 0, 255),
            'target': (255, 165, 0)
        }
        self.metrics_history: Dict[str, List[float]] = {
            'height_consistency': [],
            'horizontal_drift': [],
            'beat_timing': [],
            'dwell_ratio': [],
            'pattern_symmetry': []
        }
        self.max_history = 100
        self.graph_colors = {
            'height_consistency': (255, 165, 0),  # Orange
            'horizontal_drift': (0, 255, 0),      # Green
            'beat_timing': (255, 0, 0),          # Red
            'dwell_ratio': (0, 255, 255),        # Cyan
            'pattern_symmetry': (255, 0, 255)     # Magenta
        }
        self.stats_panel = StatisticsPanel()
        self.graph_state = GraphState()
        self.graph_interaction_region = (0, 0, 0, 0)  # x, y, width, height
        self.stats = StatsTracker()
        
    def draw_trajectories(self, frame: np.ndarray, 
                         tracked_objects: Dict[int, TrackedObject]) -> np.ndarray:
        """Draw ball trajectories with fade effect."""
        for obj in tracked_objects.values():
            if len(obj.positions) > 1:
                # Draw trajectory with fading effect
                for i in range(1, len(obj.positions)):
                    alpha = i / len(obj.positions)  # Fade based on position in trajectory
                    color = tuple([int(c * alpha) for c in self.colors['trajectory']])
                    pt1 = obj.positions[i - 1]
                    pt2 = obj.positions[i]
                    cv2.line(frame, pt1, pt2, color, 2)
                
                # Draw current position and ID
                x, y = obj.positions[-1]
                cv2.circle(frame, (x, y), 5, self.colors['ball'], -1)
                cv2.putText(frame, f"#{obj.id}", (x - 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        
        # Add speed indicators
        for obj in tracked_objects.values():
            if len(obj.positions) > 1:
                speed = self._calculate_speed(obj.positions[-2:])
                self._draw_speed_indicator(frame, obj.positions[-1], speed)
        
        # Add 3D visualization
        frame = self._draw_3d_visualization(frame, tracked_objects)
        return frame

    def draw_pattern_overlay(self, frame: np.ndarray, pattern_info: PatternInfo) -> np.ndarray:
        """Draw pattern visualization overlay."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw pattern name and confidence
        pattern_text = f"{pattern_info.pattern_type.value} ({pattern_info.confidence:.2f})"
        cv2.putText(overlay, pattern_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['pattern'], 2)
        
        # Draw pattern guide lines based on type
        if pattern_info.pattern_type.value == "Cascade":
            self._draw_cascade_guide(overlay, width, height)
        elif pattern_info.pattern_type.value == "Shower":
            self._draw_shower_guide(overlay, width, height)
        elif pattern_info.pattern_type.value == "Columns":
            self._draw_columns_guide(overlay, width, height)
            
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Update and draw heatmap
        self._update_heatmap(frame.shape, tracked_objects)
        frame = self._blend_heatmap(frame)
        
        # Draw pattern statistics
        self._draw_stats_overlay(frame, pattern_info)
        
        return frame
    
    def _draw_cascade_guide(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw guide overlay for cascade pattern."""
        center_x = width // 2
        cv2.ellipse(frame, (center_x, height//2), (width//4, height//3),
                   0, 0, 360, self.colors['pattern'], 1)
    
    def _draw_shower_guide(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw guide overlay for shower pattern."""
        points = np.array([
            [width//4, height//2],
            [3*width//4, height//2],
            [width//4, height//2]
        ], np.int32)
        cv2.polylines(frame, [points], True, self.colors['pattern'], 1)
    
    def _draw_columns_guide(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw guide overlay for columns pattern."""
        spacing = width // 6
        for x in range(spacing*2, width-spacing, spacing):
            cv2.line(frame, (x, height//4), (x, 3*height//4),
                    self.colors['pattern'], 1)
    
    def _update_heatmap(self, shape: Tuple[int, int], tracked_objects: Dict[int, TrackedObject]) -> None:
        """Update position heatmap."""
        if self.heatmap is None:
            self.heatmap = np.zeros(shape[:2], dtype=np.float32)
        
        # Fade existing heatmap
        self.heatmap *= 0.95
        
        # Add new positions
        for obj in tracked_objects.values():
            if obj.positions:
                x, y = obj.positions[-1]
                cv2.circle(self.heatmap, (x, y), 20, 1.0, -1)
    
    def _blend_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Blend heatmap with frame."""
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
    
    def _calculate_speed(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate ball speed between positions."""
        if len(positions) < 2:
            return 0.0
        p1, p2 = positions[-2:]
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _draw_speed_indicator(self, frame: np.ndarray, 
                            position: Tuple[int, int], speed: float) -> None:
        """Draw speed indicator arrow."""
        if speed > 0:
            arrow_length = min(50, int(speed))
            angle = np.random.rand() * np.pi * 2  # For demo; should use actual trajectory
            end_point = (
                int(position[0] + arrow_length * np.cos(angle)),
                int(position[1] + arrow_length * np.sin(angle))
            )
            cv2.arrowedLine(frame, position, end_point, 
                           self.colors['speed'], 2, tipLength=0.3)
    
    def _draw_stats_overlay(self, frame: np.ndarray, pattern_info: PatternInfo) -> None:
        """Draw pattern statistics overlay."""
        height, width = frame.shape[:2]
        stats_box = np.zeros((120, 200, 3), dtype=np.uint8)
        
        # Add pattern info to history
        self.stats_history.append(pattern_info.confidence)
        if len(self.stats_history) > 30:  # Keep last 30 frames
            self.stats_history.pop(0)
        
        # Draw stats background
        cv2.rectangle(frame, (10, height-130), (210, height-10), 
                     (0, 0, 0), -1)
        
        # Draw confidence history graph
        graph_points = []
        for i, conf in enumerate(self.stats_history):
            x = 20 + int(160 * i / len(self.stats_history))
            y = height - 40 - int(conf * 70)
            graph_points.append((x, y))
        
        if len(graph_points) > 1:
            cv2.polylines(frame, [np.array(graph_points)], False, 
                         self.colors['metrics'], 2)
        
        # Draw stats text
        cv2.putText(frame, f"Pattern: {pattern_info.pattern_type.value}", 
                   (20, height-100), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.colors['text'], 1)
        cv2.putText(frame, f"Confidence: {pattern_info.confidence:.2f}", 
                   (20, height-80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.colors['text'], 1)
    
    def handle_mouse_event(self, event: int, x: int, y: int, flags: int) -> None:
        """Handle mouse interactions for 3D view control."""
        # Check if mouse is in 3D view region
        vx, vy, vw, vh = self.view_3d_region
        if not (vx <= x <= vx + vw and vy <= y <= vy + vh):
            self.view_3d.is_dragging = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.view_3d.is_dragging = True
            self.view_3d.last_mouse_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.view_3d.is_dragging = False
        
        elif event == cv2.EVENT_MOUSEMOVE and self.view_3d.is_dragging:
            dx = x - self.view_3d.last_mouse_pos[0]
            dy = y - self.view_3d.last_mouse_pos[1]
            
            # Update rotation angles
            self.view_3d.rotation_y += dx * 0.5
            self.view_3d.rotation_x += dy * 0.5
            
            self.view_3d.last_mouse_pos = (x, y)
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel
            if flags > 0:
                self.view_3d.zoom *= 1.1
            else:
                self.view_3d.zoom *= 0.9

    def _project_3d_point(self, point_3d: Tuple[float, float, float],
                         center: Tuple[int, int], scale: float) -> Tuple[int, int]:
        """Project 3D point to 2D screen coordinates with rotation and zoom."""
        x, y, z = point_3d
        
        # Apply rotations
        # Rotate around X axis
        rx = radians(self.view_3d.rotation_x)
        y, z = (y * cos(rx) - z * sin(rx),
                y * sin(rx) + z * cos(rx))
        
        # Rotate around Y axis
        ry = radians(self.view_3d.rotation_y)
        x, z = (x * cos(ry) + z * sin(ry),
                -x * sin(ry) + z * cos(ry))
        
        # Apply zoom
        scale *= self.view_3d.zoom
        
        # Project to 2D
        screen_x = center[0] + int((x - z) * 0.7 * scale / 100)
        screen_y = center[1] + int((y + z * 0.5) * 0.7 * scale / 100)
        return (screen_x, screen_y)

    def _draw_3d_visualization(self, frame: np.ndarray, 
                             tracked_objects: Dict[int, TrackedObject]) -> np.ndarray:
        """Draw 3D trajectory visualization with interactive controls."""
        height, width = frame.shape[:2]
        view_size = min(width // 3, height // 3)
        offset_x = width - view_size - 10
        offset_y = 10
        
        # Update view region for mouse interaction
        self.view_3d_region = (offset_x, offset_y, view_size, view_size)
        
        # Create 3D view region
        view_region = np.zeros((view_size, view_size, 3), dtype=np.uint8)
        
        # Draw 3D grid
        self._draw_3d_grid(view_region)
        
        # Update and draw 3D trajectories
        for obj_id, obj in tracked_objects.items():
            if len(obj.positions) > 2:
                # Estimate 3D position using parabolic motion
                pos_3d = self._estimate_3d_position(obj.positions)
                
                # Update trajectory history
                if obj_id not in self.trajectory_3d_history:
                    self.trajectory_3d_history[obj_id] = []
                self.trajectory_3d_history[obj_id].append(pos_3d)
                
                # Keep limited history
                if len(self.trajectory_3d_history[obj_id]) > 30:
                    self.trajectory_3d_history[obj_id].pop(0)
                
                # Draw 3D trajectory
                self._draw_3d_trajectory(view_region, self.trajectory_3d_history[obj_id])
        
        # Blend 3D view with main frame
        frame[offset_y:offset_y+view_size, offset_x:offset_x+view_size] = \
            cv2.addWeighted(
                frame[offset_y:offset_y+view_size, offset_x:offset_x+view_size],
                0.3,
                view_region,
                0.7,
                0
            )
        
        return frame

    def _draw_3d_grid(self, view_region: np.ndarray) -> None:
        """Draw 3D coordinate grid."""
        height, width = view_region.shape[:2]
        center = (width // 2, height // 2)
        size = min(width, height) // 4
        
        # Draw coordinate axes
        cv2.line(view_region, center, 
                (center[0] + size, center[1]), 
                self.colors['3d_grid'], 1)  # X-axis
        cv2.line(view_region, center,
                (center[0], center[1] - size),
                self.colors['3d_grid'], 1)  # Y-axis
        cv2.line(view_region, center,
                (center[0] - size//2, center[1] + size//2),
                self.colors['3d_grid'], 1)  # Z-axis
        
        # Draw grid lines
        for i in range(-2, 3):
            # XZ plane
            pt1 = self._project_3d_point((i*size//2, 0, -size), center, size)
            pt2 = self._project_3d_point((i*size//2, 0, size), center, size)
            cv2.line(view_region, pt1, pt2, self.colors['3d_grid'], 1)
            
            # YZ plane
            pt1 = self._project_3d_point((0, i*size//2, -size), center, size)
            pt2 = self._project_3d_point((0, i*size//2, size), center, size)
            cv2.line(view_region, pt1, pt2, self.colors['3d_grid'], 1)

    def _estimate_3d_position(self, positions: List[Tuple[int, int]]) -> Tuple[float, float, float]:
        """Estimate 3D position using parabolic motion."""
        x, y = positions[-1]
        # Estimate z using parabolic trajectory
        t = len(positions) / 30.0  # time parameter
        z = -4.9 * t * t + 5 * t  # simplified physics
        return (float(x), float(y), z * 100)

    def _draw_3d_trajectory(self, view_region: np.ndarray,
                          trajectory: List[Tuple[float, float, float]]) -> None:
        """Draw 3D trajectory in the view region."""
        if len(trajectory) < 2:
            return
            
        height, width = view_region.shape[:2]
        center = (width // 2, height // 2)
        size = min(width, height) // 4
        
        # Draw trajectory lines
        for i in range(1, len(trajectory)):
            pt1 = self._project_3d_point(trajectory[i-1], center, size)
            pt2 = self._project_3d_point(trajectory[i], center, size)
            alpha = i / len(trajectory)
            color = tuple([int(c * alpha) for c in self.colors['3d_trajectory']])
            cv2.line(view_region, pt1, pt2, color, 2)
    
    def draw_feedback(self, frame: np.ndarray, 
                     current_metrics: AdvancedMetrics,
                     tracked_objects: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
        """Draw real-time feedback visualization."""
        # Draw ideal trajectories
        for obj_id, positions in tracked_objects.items():
            if len(positions) > 2:
                ideal_path = self._calculate_ideal_path(positions)
                self._draw_path_envelope(frame, ideal_path, self.error_threshold)
                
                # Check if current position is outside envelope
                current_pos = positions[-1]
                if self._is_outside_envelope(current_pos, ideal_path):
                    self._highlight_error(frame, current_pos)

        # Add metric indicators
        self._draw_metric_gauges(frame, current_metrics)
        return frame

    def update_drill(self, drill: DrillExercise, current_time: float):
        """Update active drill information."""
        if drill != self.drill_viz.active_drill:
            self.drill_viz.active_drill = drill
            self.drill_viz.start_time = current_time
            self.drill_viz.progress = 0.0
            self.drill_viz.success_count = 0
            self.drill_viz.attempt_count = 0

    def draw_drill_overlay(self, frame: np.ndarray, metrics: AdvancedMetrics) -> np.ndarray:
        """Draw drill visualization and instructions."""
        if not self.drill_viz.active_drill:
            return frame

        height = frame.shape[0]
        width = frame.shape[1]

        # Draw progress bar
        progress_bar_width = int(width * 0.6)
        progress_bar_height = 20
        cv2.rectangle(frame, (int(width * 0.2), height - 30), 
                     (int(width * 0.8), height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (int(width * 0.2), height - 30), 
                     (int(width * 0.2 + progress_bar_width * self.drill_viz.progress), height - 10), 
                     self.drill_colors['progress_bar'], -1)

        # Draw instructions
        instruction_text = f"Drill: {self.drill_viz.active_drill.name}"
        cv2.putText(frame, instruction_text, (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.drill_colors['instruction'], 2)

        return frame

    def _calculate_ideal_path(self, positions: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate ideal path as a line from first to last position."""
        if not positions:
            return ((0, 0), (0, 0))
        p1 = positions[0]
        p2 = positions[-1]
        return (p1, p2)

    def _draw_path_envelope(self, frame: np.ndarray, 
                          path: Tuple[Tuple[int, int], Tuple[int, int]], 
                          error_threshold: float) -> None:
        """Draw path envelope as parallel lines around ideal path."""
        p1, p2 = path
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx)
        length = np.sqrt(dx**2 + dy**2)
        
        # Normal vector
        nx = -dy / length
        ny = dx / length
        
        # Offset by error threshold
        p1_offset_high = (int(p1[0] + nx * error_threshold), int(p1[1] + ny * error_threshold))
        p2_offset_high = (int(p2[0] + nx * error_threshold), int(p2[1] + ny * error_threshold))
        p1_offset_low = (int(p1[0] - nx * error_threshold), int(p1[1] - ny * error_threshold))
        p2_offset_low = (int(p2[0] - nx * error_threshold), int(p2[1] - ny * error_threshold))
        
        # Draw envelope lines
        cv2.line(frame, p1_offset_high, p2_offset_high, (255, 0, 255), 2)
        cv2.line(frame, p1_offset_low, p2_offset_low, (255, 0, 255), 2)

    def _is_outside_envelope(self, point: Tuple[int, int], 
                           path: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if point is outside the path envelope."""
        p1, p2 = path
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return False
        
        # Normal vector
        nx = -dy / length
        ny = dx / length
        
        # Distance from point to line
        d = abs(nx * (point[0] - p1[0]) + ny * (point[1] - p1[1]))
        
        return d > self.error_threshold

    def _highlight_error(self, frame: np.ndarray, point: Tuple[int, int]) -> None:
        """Highlight error point in the frame."""
        cv2.circle(frame, point, 8, (0, 0, 255), -1)

    def _draw_metric_gauges(self, frame: np.ndarray, metrics: AdvancedMetrics) -> None:
        """Draw gauges for advanced metrics."""
        height, width = frame.shape[:2]
        gauge_width = 100
        gauge_height = 10
        
        # Draw each metric gauge
        for i, (metric_name, value) in enumerate(metrics.__dict__.items()):
            if metric_name in self.graph_colors:
                # Position for the gauge
                x = 10
                y = height - 30 - i * (gauge_height + 10)
                
                # Draw gauge background
                cv2.rectangle(frame, (x, y), (x + gauge_width, y + gauge_height), (50, 50, 50), -1)
                
                # Draw gauge fill
                fill_width = int(gauge_width * value)
                cv2.rectangle(frame, (x, y), (x + fill_width, y + gauge_height), 
                             self.graph_colors[metric_name], -1)
                
                # Draw metric name
                cv2.putText(frame, metric_name.replace('_', ' ').capitalize(), 
                           (x + 5, y + gauge_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (255, 255, 255), 1)
        
        return frame

    def add_heatmap(self, frame: np.ndarray, tracked_objects: Dict[int, TrackedObject]) -> np.ndarray:
        """Add position heatmap to the frame."""
        if self.heatmap is None:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        # Accumulate heatmap data
        for obj in tracked_objects.values():
            if obj.positions:
                for pos in obj.positions:
                    cv2.circle(self.heatmap, pos, 5, 1.0, -1)
        
        # Normalize and apply colormap
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with frame
        frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        return frame

    def reset_heatmap(self):
        """Reset the position heatmap."""
        self.heatmap = None

    def update_statistics(self, metrics: AdvancedMetrics):
        """Update statistics panel with new metrics."""
        self.metrics_history['height_consistency'].append(metrics.throw_height_consistency)
        self.metrics_history['horizontal_drift'].append(metrics.horizontal_drift)
        self.metrics_history['beat_timing'].append(metrics.beat_timing_error)
        self.metrics_history['dwell_ratio'].append(metrics.dwell_ratio)
        self.metrics_history['pattern_symmetry'].append(metrics.pattern_symmetry)
        
        # Limit history size
        for key in self.metrics_history.keys():
            if len(self.metrics_history[key]) > self.max_history:
                self.metrics_history[key].pop(0)

    def draw_statistics_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw the statistics panel on the frame."""
        height, width = frame.shape[:2]
        panel_height = 150
        graph_width = 300
        graph_height = 100
        
        # Draw panel background
        cv2.rectangle(frame, (10, 10), (310, 10 + panel_height), (0, 0, 0), -1)
        
        # Draw each metric graph
        for i, (metric_name, color) in enumerate(self.graph_colors.items()):
            if metric_name in self.metrics_history:
                # Prepare graph points
                points = self.metrics_history[metric_name]
                if len(points) > 1:
                    # Normalize points for graph
                    max_val = max(points)
                    min_val = min(points)
                    range_val = max_val - min_val if max_val - min_val > 0 else 1
                    graph_points = [
                        (int(10 + j * (graph_width - 20) / (len(points) - 1)), 
                         int(10 + panel_height - (p - min_val) * graph_height / range_val))
                        for j, p in enumerate(points)
                    ]
                    
                    # Draw graph line
                    cv2.polylines(frame, [np.array(graph_points)], False, color, 2)
                
                # Draw metric name
                cv2.putText(frame, metric_name.replace('_', ' ').capitalize(), 
                           (320, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        return frame

    def draw_statistics_overlay(self, frame: np.ndarray, metrics: AdvancedMetrics) -> np.ndarray:
        """Draw interactive statistics visualization overlay."""
        height, width = frame.shape[:2]
        
        # Update statistics
        self.stats_panel.update(metrics)
        
        # Create stats panel region
        panel_width = width // 4
        panel_height = height
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Update graph interaction region
        self.graph_interaction_region = (
            width - panel_width, 0, panel_width, panel_height
        )
        
        # Draw metrics graphs
        graph_height = panel_height // 6
        visible_metrics = [m for m, v in self.graph_state.visible_metrics.items() if v]
        
        for i, metric in enumerate(visible_metrics):
            values = self.stats_panel.metrics_history[metric]
            if not values:
                continue
            
            # Calculate visible range
            start_idx = max(0, len(values) - self.graph_state.time_window - self.graph_state.pan_offset)
            end_idx = min(len(values), start_idx + self.graph_state.time_window)
            visible_values = values[start_idx:end_idx]
            
            # Draw graph background and border
            y_offset = i * graph_height + 10
            cv2.rectangle(panel,
                         (10, y_offset),
                         (panel_width-10, y_offset+graph_height-10),
                         (50, 50, 50), 1)
            
            # Draw metric name with toggle indicator
            color = self.stats_panel.graph_colors[metric]
            toggle_char = "◉" if self.graph_state.visible_metrics[metric] else "○"
            cv2.putText(panel, f"{toggle_char} {metric.replace('_', ' ').title()}",
                       (15, y_offset+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       color, 1)
            
            # Draw graph line
            if len(visible_values) > 1:
                points = []
                for j, value in enumerate(visible_values):
                    x = 10 + int((panel_width-20) * j / len(visible_values))
                    y = y_offset + graph_height - 20 - int((graph_height-30) * value)
                    points.append((x, y))
                cv2.polylines(panel, [np.array(points)], False, color, 1)
            
            # Draw current value
            if values:
                cv2.putText(panel, f"{values[-1]:.2f}",
                           (panel_width-60, y_offset+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           color, 1)

        # Draw zoom level indicator
        cv2.putText(panel, f"Window: {self.graph_state.time_window}f",
                   (10, panel_height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (200, 200, 200), 1)

        # Blend panel with frame
        frame[:, -panel_width:] = cv2.addWeighted(
            frame[:, -panel_width:], 0.3,
            panel, 0.7,
            0
        )
        
        return frame

    def handle_graph_interaction(self, event: int, x: int, y: int, flags: int) -> None:
        """Handle mouse interactions for graph controls."""
        # Check if mouse is in graph region
        gx, gy, gw, gh = self.graph_interaction_region
        if not (gx <= x <= gx + gw and gy <= y <= gy + gh):
            self.graph_state.is_dragging = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on metric label for toggling
            self._handle_metric_toggle(x, y)
            self.graph_state.is_dragging = True
            self.graph_state.last_mouse_pos = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.graph_state.is_dragging = False
            
        elif event == cv2.EVENT_MOUSEMOVE and self.graph_state.is_dragging:
            # Pan the view
            dx = x - self.graph_state.last_mouse_pos[0]
            self.graph_state.pan_offset += dx
            self.graph_state.pan_offset = max(0, min(
                self.graph_state.pan_offset,
                len(next(iter(self.metrics_history.values()))) - self.graph_state.time_window
            ))
            self.graph_state.last_mouse_pos = (x, y)
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:
                self.graph_state.time_window = max(20, self.graph_state.time_window - 10)
            else:
                self.graph_state.time_window = min(200, self.graph_state.time_window + 10)

    def _handle_metric_toggle(self, x: int, y: int) -> None:
        """Toggle metric visibility when its label is clicked."""
        panel_width = self.graph_interaction_region[2]
        for i, metric in enumerate(self.graph_state.visible_metrics):
            label_y = self.graph_interaction_region[1] + i * 20
            if y >= label_y and y <= label_y + 15:
                self.graph_state.visible_metrics[metric] = not self.graph_state.visible_metrics[metric]
                break

class GraphPanel:
    def handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        try:
            if not hasattr(self, 'state'):
                return
                
            # Convert coordinates to local space
            local_x = x
            local_y = y
            
            if event == cv2.EVENT_MOUSEWHEEL:
                zoom_factor = 1.1 if flags > 0 else 0.9
                self.state.zoom_level = max(0.1, min(5.0, self.state.zoom_level * zoom_factor))
                
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.state.is_dragging = True
                self.state.last_mouse_pos = (local_x, local_y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.state.is_dragging = False
                
            elif event == cv2.EVENT_MOUSEMOVE and self.state.is_dragging:
                dx = local_x - self.state.last_mouse_pos[0]
                self.state.pan_offset = max(0, min(
                    self.state.pan_offset - dx,
                    self._get_max_pan()
                ))
                self.state.last_mouse_pos = (local_x, local_y)
                
        except Exception as e:
            print(f"Error in mouse handling: {e}")