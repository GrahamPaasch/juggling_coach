import cv2
from math import cos, sin, radians
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from ..tracking.tracker import TrackedObject
from ..analysis.pattern_recognition import PatternInfo

@dataclass
class View3DState:
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    zoom: float = 1.0
    is_dragging: bool = False
    last_mouse_pos: Tuple[int, int] = (0, 0)

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