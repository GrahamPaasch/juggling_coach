import cv2
import numpy as np
from typing import Tuple, Dict, Optional

class TouchUI:
    def __init__(self, screen_width: int = 320, screen_height: int = 240):
        self.width = screen_width
        self.height = screen_height
        self.button_height = 40
        self.button_states: Dict[str, bool] = {
            'calibrate': False,
            'settings': False,
            'color_picker': False
        }
        self.active_color = (30, 50, 50)  # HSV
        
    def handle_touch(self, x: int, y: int) -> str:
        """Handle touch events and return action."""
        if y < self.button_height:
            # Handle button touches
            button_width = self.width // 3
            if x < button_width:
                return 'calibrate'
            elif x < button_width * 2:
                return 'settings'
            else:
                return 'color_picker'
        return ''
    
    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """Draw touch controls on frame."""
        # Draw button bar
        button_width = self.width // 3
        
        # Calibrate button
        cv2.rectangle(frame, (0, 0), (button_width, self.button_height),
                     (0, 255, 0) if self.button_states['calibrate'] else (0, 200, 0), -1)
        cv2.putText(frame, "Calibrate", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Settings button
        cv2.rectangle(frame, (button_width, 0), (button_width * 2, self.button_height),
                     (0, 200, 200) if self.button_states['settings'] else (0, 150, 150), -1)
        cv2.putText(frame, "Settings", (button_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Color picker button
        cv2.rectangle(frame, (button_width * 2, 0), (self.width, self.button_height),
                     (0, 0, 255) if self.button_states['color_picker'] else (0, 0, 200), -1)
        cv2.putText(frame, "Color", (button_width * 2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def show_color_picker(self, frame: np.ndarray) -> np.ndarray:
        """Show color picker overlay."""
        if not self.button_states['color_picker']:
            return frame
            
        # Draw color picker overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 60), (self.width-40, self.height-40),
                     (0, 0, 0), -1)
        
        # Draw HSV color sliders
        for i, (name, value) in enumerate(zip(['H', 'S', 'V'], self.active_color)):
            y_pos = 100 + i * 50
            cv2.putText(overlay, name, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(overlay, (70, y_pos-10),
                         (self.width-60, y_pos+10), (200, 200, 200), -1)
            marker_x = 70 + int((value/255) * (self.width-130))
            cv2.circle(overlay, (marker_x, y_pos), 10, (0, 255, 0), -1)
        
        # Blend overlay with frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame