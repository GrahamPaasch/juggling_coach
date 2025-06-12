import cv2
import numpy as np

class BallDetector:
    def __init__(self, config):
        self.config = config
        # Convert color bounds to numpy arrays
        self.color_lower = np.array(config.color_lower)
        self.color_upper = np.array(config.color_upper)
        self.min_radius = config.min_ball_radius
        self.max_radius = config.max_ball_radius
    
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use class attributes instead of dictionary access
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if self.min_radius <= radius <= self.max_radius:
                centers.append((int(x), int(y)))
        
        return centers