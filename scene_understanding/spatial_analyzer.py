"""
Spatial Analyzer - Analyzes object positions and spatial relationships
Provides distance estimation, direction, and spatial context
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Direction(Enum):
    """Spatial directions relative to user"""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    FAR_LEFT = "far left"
    FAR_RIGHT = "far right"


class DistanceZone(Enum):
    """Distance zones for objects"""
    IMMEDIATE = "immediate"  # < 2m
    NEAR = "near"           # 2-5m
    FAR = "far"             # > 5m


class SpatialInfo:
    """Spatial information for a detected object"""
    
    def __init__(self, detection, direction: Direction, distance: Optional[float],
                 distance_zone: DistanceZone, relative_size: float, is_moving: bool = False):
        self.detection = detection
        self.direction = direction
        self.distance = distance  # in meters (if available)
        self.distance_zone = distance_zone
        self.relative_size = relative_size  # 0-1, proportion of frame
        self.is_moving = is_moving
        self.velocity = None  # Will be set by motion analyzer
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'class_name': self.detection.class_name,
            'direction': self.direction.value,
            'distance': self.distance,
            'distance_zone': self.distance_zone.value,
            'relative_size': self.relative_size,
            'is_moving': self.is_moving,
            'confidence': self.detection.confidence
        }
    
    def __repr__(self) -> str:
        dist_str = f"{self.distance:.1f}m" if self.distance else self.distance_zone.value
        return f"Spatial({self.detection.class_name} @ {self.direction.value}, {dist_str})"


class SpatialAnalyzer:
    """Analyzes spatial relationships of detected objects"""
    
    def __init__(self, camera_fov: float = 62.0, camera_height: float = 1.5,
                 frame_width: int = 640, frame_height: int = 480):
        """
        Initialize spatial analyzer
        
        Args:
            camera_fov: Camera field of view in degrees (default 62° for Pi Camera V2)
            camera_height: Camera mounting height in meters (default 1.5m, chest level)
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
        """
        self.camera_fov = camera_fov
        self.camera_height = camera_height
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define spatial zones (as fraction of frame width)
        self.zone_boundaries = {
            'far_left': (0.0, 0.2),
            'left': (0.2, 0.4),
            'center': (0.4, 0.6),
            'right': (0.6, 0.8),
            'far_right': (0.8, 1.0)
        }
        
        # Known object heights for distance estimation (in meters)
        self.object_heights = {
            'person': 1.7,
            'car': 1.5,
            'truck': 2.5,
            'bus': 3.0,
            'bicycle': 1.1,
            'motorcycle': 1.2,
            'traffic light': 3.5,
            'stop sign': 2.5,
            'fire hydrant': 0.7,
            'bench': 0.5,
            'chair': 0.9,
            'dog': 0.6,
            'cat': 0.3
        }
        
        # Calculate focal length (in pixels)
        self.focal_length = (frame_width / 2) / np.tan(np.radians(camera_fov / 2))
        
    def analyze(self, detections: List) -> List[SpatialInfo]:
        """
        Analyze spatial information for all detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of SpatialInfo objects
        """
        spatial_infos = []
        
        for det in detections:
            direction = self._get_direction(det)
            distance = self._estimate_distance(det)
            distance_zone = self._get_distance_zone(distance, det)
            relative_size = self._get_relative_size(det)
            
            spatial_info = SpatialInfo(
                detection=det,
                direction=direction,
                distance=distance,
                distance_zone=distance_zone,
                relative_size=relative_size
            )
            
            spatial_infos.append(spatial_info)
        
        return spatial_infos
    
    def _get_direction(self, detection) -> Direction:
        """Determine direction of object relative to camera center"""
        center_x, _ = detection.center
        normalized_x = center_x / self.frame_width
        
        for zone_name, (lower, upper) in self.zone_boundaries.items():
            if lower <= normalized_x < upper:
                if zone_name == 'far_left':
                    return Direction.FAR_LEFT
                elif zone_name == 'left':
                    return Direction.LEFT
                elif zone_name == 'center':
                    return Direction.CENTER
                elif zone_name == 'right':
                    return Direction.RIGHT
                elif zone_name == 'far_right':
                    return Direction.FAR_RIGHT
        
        return Direction.CENTER
    
    def _estimate_distance(self, detection) -> Optional[float]:
        """
        Estimate distance to object using similar triangles
        Distance = (Real Height * Focal Length) / Pixel Height
        
        Args:
            detection: Detection object
            
        Returns:
            Estimated distance in meters, or None if cannot estimate
        """
        class_name = detection.class_name
        
        # Check if we have known height for this object
        if class_name not in self.object_heights:
            return None
        
        real_height = self.object_heights[class_name]
        pixel_height = detection.height
        
        if pixel_height == 0:
            return None
        
        # Calculate distance
        distance = (real_height * self.focal_length) / pixel_height
        
        # Sanity check: distances should be reasonable (0.5m to 50m)
        if 0.5 <= distance <= 50.0:
            return distance
        
        return None
    
    def _get_distance_zone(self, distance: Optional[float], detection) -> DistanceZone:
        """
        Determine distance zone
        If distance estimation is unavailable, use relative size heuristic
        """
        if distance is not None:
            if distance < 2.0:
                return DistanceZone.IMMEDIATE
            elif distance < 5.0:
                return DistanceZone.NEAR
            else:
                return DistanceZone.FAR
        
        # Fallback: use relative size
        relative_size = self._get_relative_size(detection)
        
        if relative_size > 0.3:  # Object takes up > 30% of frame
            return DistanceZone.IMMEDIATE
        elif relative_size > 0.1:  # Object takes up > 10% of frame
            return DistanceZone.NEAR
        else:
            return DistanceZone.FAR
    
    def _get_relative_size(self, detection) -> float:
        """
        Calculate relative size of object (proportion of frame)
        
        Returns:
            Float between 0 and 1
        """
        bbox_area = detection.area
        frame_area = self.frame_width * self.frame_height
        
        return bbox_area / frame_area
    
    def get_nearest_objects(self, spatial_infos: List[SpatialInfo], n: int = 3) -> List[SpatialInfo]:
        """
        Get N nearest objects
        
        Args:
            spatial_infos: List of SpatialInfo objects
            n: Number of nearest objects to return
            
        Returns:
            List of nearest SpatialInfo objects
        """
        # Sort by distance (None values go to end)
        sorted_infos = sorted(
            spatial_infos,
            key=lambda x: (x.distance is None, x.distance if x.distance else float('inf'))
        )
        
        return sorted_infos[:n]
    
    def get_objects_in_direction(self, spatial_infos: List[SpatialInfo], 
                                 direction: Direction) -> List[SpatialInfo]:
        """Get all objects in a specific direction"""
        return [si for si in spatial_infos if si.direction == direction]
    
    def get_objects_in_zone(self, spatial_infos: List[SpatialInfo],
                           zone: DistanceZone) -> List[SpatialInfo]:
        """Get all objects in a specific distance zone"""
        return [si for si in spatial_infos if si.distance_zone == zone]
    
    def calculate_angle_to_object(self, detection) -> float:
        """
        Calculate horizontal angle to object center (in degrees)
        Negative = left, Positive = right, 0 = center
        
        Args:
            detection: Detection object
            
        Returns:
            Angle in degrees
        """
        center_x, _ = detection.center
        frame_center_x = self.frame_width / 2
        
        # Calculate pixel offset from center
        pixel_offset = center_x - frame_center_x
        
        # Convert to angle using FOV
        angle = (pixel_offset / frame_center_x) * (self.camera_fov / 2)
        
        return angle
    
    def is_object_blocking_path(self, spatial_info: SpatialInfo, 
                                path_width: float = 1.0) -> bool:
        """
        Check if object is blocking the forward path
        
        Args:
            spatial_info: SpatialInfo object
            path_width: Width of path to check in meters (default 1m)
            
        Returns:
            True if object is blocking path
        """
        # Check if object is in center zone
        if spatial_info.direction != Direction.CENTER:
            return False
        
        # Check if object is close enough to matter
        if spatial_info.distance_zone == DistanceZone.FAR:
            return False
        
        # Calculate if object is within path width
        angle = self.calculate_angle_to_object(spatial_info.detection)
        
        if spatial_info.distance:
            # Calculate lateral distance
            lateral_distance = spatial_info.distance * np.tan(np.radians(abs(angle)))
            return lateral_distance < path_width / 2
        
        # If no distance, use conservative approach
        return spatial_info.distance_zone == DistanceZone.IMMEDIATE