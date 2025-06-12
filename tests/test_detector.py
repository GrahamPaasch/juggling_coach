import pytest
import cv2
import numpy as np
from src.tracking.detector import BallDetector
from src.utils.config import TrackingConfig

@pytest.fixture
def config():
    return TrackingConfig(
        min_ball_radius=10,
        max_ball_radius=30,
        color_lower=(30, 50, 50),
        color_upper=(80, 255, 255),
        max_disappeared=30
    )

@pytest.fixture
def detector(config):
    return BallDetector(config)

def test_detector_initialization(detector, config):
    """Test detector initialization and color conversion."""
    assert np.array_equal(detector.color_lower, np.array(config.color_lower))
    assert np.array_equal(detector.color_upper, np.array(config.color_upper))

def test_empty_frame_detection(detector):
    """Test detection on empty frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    centers = detector.detect(frame)
    assert len(centers) == 0

def test_detect_single_ball(detector):
    """Test detection of a single green ball."""
    # Create test frame with a green ball
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a green ball in HSV space
    frame = cv2.circle(frame, (100, 100), 15, (60, 255, 255), -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    centers = detector.detect(frame)
    
    assert len(centers) == 1
    assert abs(centers[0][0] - 100) < 5
    assert abs(centers[0][1] - 100) < 5

def test_detect_multiple_balls(detector):
    """Test detection of multiple balls."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    positions = [(50, 50), (150, 150)]
    
    for x, y in positions:
        frame = cv2.circle(frame, (x, y), 15, (60, 255, 255), -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    centers = detector.detect(frame)
    
    assert len(centers) == 2
    for (x, y), (cx, cy) in zip(positions, centers):
        assert abs(x - cx) < 5
        assert abs(y - cy) < 5

def test_size_filtering(detector):
    """Test ball size filtering."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw balls of different sizes
    frame = cv2.circle(frame, (50, 50), 5, (60, 255, 255), -1)  # Too small
    frame = cv2.circle(frame, (150, 150), 40, (60, 255, 255), -1)  # Too big
    frame = cv2.circle(frame, (100, 100), 15, (60, 255, 255), -1)  # Just right
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    centers = detector.detect(frame)
    
    assert len(centers) == 1
    assert abs(centers[0][0] - 100) < 5
    assert abs(centers[0][1] - 100) < 5

def test_no_balls_detected(detector):
    """Test detection with no balls in frame."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    centers = detector.detect(img)
    assert len(centers) == 0

@pytest.mark.parametrize("ball_color,expected_detection", [
    ((60, 255, 255), True),   # Green ball - should detect
    ((0, 255, 255), False),   # Red ball - should not detect
    ((30, 50, 50), False),    # Dark green - edge case
    ((80, 255, 255), False)   # Blue - should not detect
])
def test_color_detection(detector, ball_color, expected_detection):
    """Test ball detection with different colors."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 20, ball_color, -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    centers = detector.detect(img)
    assert (len(centers) > 0) == expected_detection

@pytest.mark.parametrize("radius", [5, 10, 20, 35])
def test_ball_size_detection(detector, radius):
    """Test detection with different ball sizes."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), radius, (60, 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    centers = detector.detect(img)
    expected = 10 <= radius <= 30  # Should detect only balls within size range
    assert (len(centers) > 0) == expected

@pytest.mark.parametrize("noise_level", [0, 0.1, 0.3, 0.5])
def test_noise_robustness(detector, noise_level):
    """Test detection with different noise levels."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 20, (60, 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    # Add noise
    noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    
    centers = detector.detect(noisy_img)
    if noise_level < 0.3:  # Should detect ball with reasonable noise
        assert len(centers) == 1
        assert abs(centers[0][0] - 150) < 10
        assert abs(centers[0][1] - 150) < 10