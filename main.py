import cv2
from pathlib import Path
from src.utils.config import Config
from src.tracking.detector import BallDetector
from src.tracking.tracker import CentroidTracker
from src.analysis.metrics import PatternAnalyzer
from src.analysis.pattern_recognition import PatternRecognizer
from src.visualization.display import PatternVisualizer

def main():
    # Load configuration
    config_path = Path('config/settings.yaml')
    config = Config.load(config_path)
    
    # Initialize components
    detector = BallDetector(config.tracking)
    tracker = CentroidTracker(max_disappeared=config.tracking.max_disappeared)
    analyzer = PatternAnalyzer()
    pattern_recognizer = PatternRecognizer()
    visualizer = PatternVisualizer()
    
    # Set up mouse callback
    def mouse_callback(event, x, y, flags, param):
        visualizer.handle_mouse_event(event, x, y, flags)
    
    cv2.namedWindow('Juggling Coach')
    cv2.setMouseCallback('Juggling Coach', mouse_callback)
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect and track balls
        ball_positions = detector.detect(frame)
        tracked_objects = tracker.update(ball_positions)
        
        # Draw trajectories
        frame = visualizer.draw_trajectories(frame, tracked_objects)
        
        # Analyze pattern and draw visualization
        pattern_info = pattern_recognizer.recognize_pattern(tracked_objects)
        if pattern_info:
            frame = visualizer.draw_pattern_overlay(frame, pattern_info)
        
        cv2.imshow('Juggling Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()