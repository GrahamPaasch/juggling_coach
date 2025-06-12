import cv2
import numpy as np
from pathlib import Path
from src.tracking.detector import BallDetector
from src.tracking.tracker import CentroidTracker
from src.ui.touch_controls import TouchUI

class JugglingCoachMobile:
    def __init__(self):
        # Simple default config for mobile
        self.config = {
            'min_ball_radius': 10,
            'max_ball_radius': 30,
            'color_lower': np.array([30, 50, 50]),
            'color_upper': np.array([80, 255, 255]),
            'max_disappeared': 30
        }
        
        # Initialize components
        self.detector = BallDetector(self.config)
        self.tracker = CentroidTracker(max_disappeared=self.config['max_disappeared'])
        self.ui = TouchUI(320, 240)
        
    def handle_touch(self, event, x, y, flags, param):
        """Handle touch events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            action = self.ui.handle_touch(x, y)
            if action == 'calibrate':
                self.ui.button_states['calibrate'] = True
                # Start calibration mode
            elif action == 'settings':
                self.ui.button_states['settings'] = True
                # Show settings panel
            elif action == 'color_picker':
                self.ui.button_states['color_picker'] = not self.ui.button_states['color_picker']
    
    def process_frame(self, frame):
        """Process a single frame and return visualization."""
        # Resize frame for mobile performance
        frame = cv2.resize(frame, (320, 240))
        
        # Detect and track balls
        ball_positions = self.detector.detect(frame)
        tracked_objects = self.tracker.update(ball_positions)
        
        # Draw results
        for obj_id, obj in tracked_objects.items():
            if obj.positions:
                x, y = obj.positions[-1]
                # Draw circle and ID
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"#{obj_id}", (x - 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw trajectory
                if len(obj.positions) > 1:
                    for i in range(1, len(obj.positions)):
                        pt1 = obj.positions[i - 1]
                        pt2 = obj.positions[i]
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        
        # Add UI elements
        frame = self.ui.draw_controls(frame)
        if self.ui.button_states['color_picker']:
            frame = self.ui.show_color_picker(frame)
        
        return frame

def main():
    app = JugglingCoachMobile()
    camera = cv2.VideoCapture(0)
    
    def touch_callback(event, x, y, flags, param):
        app.handle_touch(event, x, y, flags, param)
    
    cv2.namedWindow("Juggling Coach Mobile")
    cv2.setMouseCallback("Juggling Coach Mobile", touch_callback)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = app.process_frame(frame)
        
        # Display result
        if processed_frame is not None and isinstance(processed_frame, np.ndarray):
            cv2.imshow("Juggling Coach Mobile", processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()