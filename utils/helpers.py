import numpy as np

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union between bounding boxes"""
    x1_i = max(bbox1[0], bbox2[0])
    y1_i = max(bbox1[1], bbox2[1])
    x2_i = min(bbox1[2], bbox2[2])
    y2_i = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    return safe_divide(inter_area, union_area)

def calculate_distance(point1, point2):
    """Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def normalize_bbox(bbox, img_width, img_height):
    """Normalize bbox coordinates to 0-1 range"""
    x1, y1, x2, y2 = bbox
    return [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def moving_average(data, window_size=5):
    """Calculate moving average of data series"""
    if len(data) < window_size:
        return np.mean(data) if data else 0
    return np.mean(data[-window_size:])

def exponential_smoothing(data, alpha=0.3):
    """Apply exponential smoothing to data"""
    if not data:
        return []
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def calculate_angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.degrees(np.arccos(safe_divide(dot_product, norm_v1 * norm_v2)))

def format_distance(meters):
    """
    Format distance for speech
    
    Args:
        meters: Distance in meters
    
    Returns:
        str: "X meters" or "X centimeters" or "very close"
    """
    if meters < 0.5:
        return "very close"
    if meters < 1:
        return f"{int(meters * 100)} centimeters"
    return f"{meters:.1f} meters"

def format_time(seconds):
    """Format seconds for speech ("5 seconds", "1 minute")"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = int(seconds / 60)
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"

def safe_divide(numerator, denominator, default=0):
    """Division with zero handling"""
    return numerator / denominator if denominator != 0 else default
