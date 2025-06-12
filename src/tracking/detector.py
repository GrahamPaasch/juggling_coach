import cv2
import numpy as np

class BallDetector:
    def __init__(self, config):
        self.config = config
    
    def detect(self, frame):
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.config['color_lower'], self.config['color_upper'])
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)  # Reduced iterations for performance
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if self.config['min_ball_radius'] <= radius <= self.config['max_ball_radius']:
                centers.append((int(x), int(y)))
        
        return centers