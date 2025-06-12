import cv2
from pathlib import Path
from src.utils.config import Config
from src.tracking.detector import BallDetector
from src.tracking.tracker import CentroidTracker
from src.analysis.metrics import PatternAnalyzer
from src.analysis.pattern_recognition import PatternRecognizer
from src.visualization.display import PatternVisualizer
from src.training.drill_generator import DrillGenerator, DrillConfig
from src.visualization.graph_panel import GraphPanel  # Import GraphPanel

def main():
    # Load configuration
    config_path = Path('config/settings.yaml')
    config = Config.load(config_path)
    
    # Initialize components
    detector = BallDetector(config.tracking)
    tracker = CentroidTracker(max_disappeared=config.tracking.max_disappeared)
    visualizer = PatternVisualizer()
    graph_panel = GraphPanel(width=320, height=720)
    
    # Set up mouse callback
    def mouse_callback(event, x, y, flags, param):
        visualizer.handle_mouse_event(event, x, y, flags)  # For 3D view
        visualizer.handle_graph_interaction(event, x, y, flags)  # For graphs
        if x > frame.shape[1] - graph_panel.width:
            # Adjust x coordinate relative to graph panel
            graph_panel.handle_mouse(event, x - (frame.shape[1] - graph_panel.width), y, flags)
    
    cv2.namedWindow('Juggling Coach')
    cv2.setMouseCallback('Juggling Coach', mouse_callback)
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect and track balls
        ball_positions = detector.detect(frame)
        tracked_objects = tracker.update(ball_positions)
        
        # Calculate metrics
        metrics = analyzer.calculate_advanced_metrics(tracked_objects)
        pattern_info = pattern_recognizer.recognize_pattern(tracked_objects)
        
        # Update drill progress
        if drill_generator.current_drill:
            progress = drill_generator.update_progress(metrics)
            visualizer.drill_viz.progress = progress
            
            # Check if drill is complete
            if progress >= 1.0:
                if drill_generator.evaluate_progress(metrics):
                    print("\nDrill completed successfully!")
                    drill_generator.complete_drill(metrics)  # Save statistics
                else:
                    print("\nDrill needs more practice.")
                drill_generator.current_drill = None
        
        # Generate new drill if needed
        if not drill_generator.current_drill:
            drill = drill_generator.generate_drill(metrics, pattern_info)
            visualizer.update_drill(drill, time.time())
            drill_generator.current_drill = drill
            drill_generator.drill_start_time = time.time()
            print(f"\nNew Drill: {drill.name}")
            print(f"Focus: {drill.focus_metric}")
            for instruction in drill.instructions:
                print(instruction)
        
        # Update metrics
        metrics_dict = {
            'height_consistency': metrics.throw_height_consistency,
            'horizontal_drift': metrics.horizontal_drift,
            'beat_timing': metrics.beat_timing_error,
            'dwell_ratio': metrics.dwell_ratio,
            'pattern_symmetry': metrics.pattern_symmetry
        }
        graph_panel.stats.update_metrics(metrics_dict)
        
        # Draw visualizations
        frame = visualizer.draw_trajectories(frame, tracked_objects)
        if pattern_info:
            frame = visualizer.draw_pattern_overlay(frame, pattern_info)
        frame = visualizer.draw_drill_overlay(frame, metrics)
        frame = visualizer.draw_statistics_overlay(frame, metrics)  # Add statistics visualization
        frame = graph_panel.render(frame)
        
        cv2.imshow('Juggling Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
        # Save periodic metrics
        if frame_count % (fps * 30) == 0:  # Every 30 seconds
            drill_generator.stats_manager.save_session_metrics(metrics)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()