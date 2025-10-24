"""
Object priority rules for blind assistance system
Defines importance levels and announcement strategies for detected objects
"""

from enum import Enum
from typing import Dict, List, Set


class Priority(Enum):
    """Priority levels for detected objects"""
    CRITICAL = 4    # Immediate danger - announce immediately
    HIGH = 3        # Important obstacles - announce with short delay
    MEDIUM = 2      # Relevant objects - announce if persistent
    LOW = 1         # Background objects - announce only on request
    IGNORE = 0      # Non-relevant objects - don't announce


class ObjectPriorityRules:
    """Defines priority rules for different object types"""
    
    # Priority mappings for COCO classes
    PRIORITY_MAP: Dict[str, Priority] = {
        # CRITICAL - Moving vehicles and immediate hazards
        "car": Priority.CRITICAL,
        "truck": Priority.CRITICAL,
        "bus": Priority.CRITICAL,
        "motorcycle": Priority.CRITICAL,
        "bicycle": Priority.HIGH,
        "train": Priority.CRITICAL,
        
        # HIGH - Obstacles and navigation hazards
        "person": Priority.HIGH,
        "fire hydrant": Priority.HIGH,
        "stop sign": Priority.HIGH,
        "traffic light": Priority.HIGH,
        "bench": Priority.MEDIUM,
        "parking meter": Priority.MEDIUM,
        
        # MEDIUM - Relevant environmental objects
        "chair": Priority.MEDIUM,
        "couch": Priority.MEDIUM,
        "potted plant": Priority.MEDIUM,
        "dining table": Priority.MEDIUM,
        "door": Priority.MEDIUM,
        "stairs": Priority.HIGH,
        
        # LOW - Background/informational objects
        "book": Priority.LOW,
        "clock": Priority.LOW,
        "vase": Priority.LOW,
        "tv": Priority.LOW,
        "laptop": Priority.LOW,
        "cell phone": Priority.LOW,
        "cup": Priority.LOW,
        "bottle": Priority.LOW,
        
        # Animals - context dependent
        "dog": Priority.HIGH,
        "cat": Priority.MEDIUM,
        "bird": Priority.LOW,
        "horse": Priority.HIGH,
    }
    
    # Distance-based priority modifiers (in meters)
    DISTANCE_THRESHOLDS = {
        Priority.CRITICAL: {
            'immediate': 2.0,   # Within 2m - announce immediately
            'near': 5.0,        # Within 5m - announce soon
            'far': 10.0         # Beyond 10m - lower priority
        },
        Priority.HIGH: {
            'immediate': 1.5,
            'near': 3.0,
            'far': 6.0
        },
        Priority.MEDIUM: {
            'immediate': 1.0,
            'near': 2.0,
            'far': 4.0
        }
    }
    
    # Cooldown periods (seconds) - how long to wait before re-announcing same object
    COOLDOWN_PERIODS = {
        Priority.CRITICAL: 3.0,   # Re-announce every 3 seconds if still present
        Priority.HIGH: 5.0,       # Re-announce every 5 seconds
        Priority.MEDIUM: 10.0,    # Re-announce every 10 seconds
        Priority.LOW: 30.0        # Re-announce every 30 seconds
    }
    
    # Movement sensitivity - speed threshold (m/s) to trigger announcement
    MOVEMENT_THRESHOLDS = {
        Priority.CRITICAL: 0.3,   # Any movement > 0.3 m/s
        Priority.HIGH: 0.5,       # Movement > 0.5 m/s
        Priority.MEDIUM: 1.0,     # Fast movement only
        Priority.LOW: float('inf') # Don't announce based on movement
    }
    
    # Spatial zones - which areas matter most
    SPATIAL_ZONES = {
        'center': 1.5,      # Multiplier for objects in center of view
        'left': 1.2,        # Multiplier for objects on sides
        'right': 1.2,
        'peripheral': 0.8   # Lower priority for peripheral objects
    }
    
    # Objects that should always be tracked for movement
    TRACK_MOVEMENT: Set[str] = {
        "car", "truck", "bus", "motorcycle", "bicycle", 
        "person", "dog", "train"
    }
    
    # Objects that indicate specific contexts
    CONTEXT_INDICATORS = {
        'road_crossing': ['car', 'truck', 'bus', 'traffic light', 'stop sign'],
        'indoor': ['chair', 'couch', 'tv', 'dining table', 'bed'],
        'outdoor': ['bicycle', 'bench', 'fire hydrant', 'traffic light'],
        'stairs_area': ['stairs', 'handrail'],
    }
    
    @classmethod
    def get_priority(cls, class_name: str) -> Priority:
        """Get priority for a given object class"""
        return cls.PRIORITY_MAP.get(class_name, Priority.LOW)
    
    @classmethod
    def should_track_movement(cls, class_name: str) -> bool:
        """Check if object movement should be tracked"""
        return class_name in cls.TRACK_MOVEMENT
    
    @classmethod
    def get_cooldown(cls, priority: Priority) -> float:
        """Get cooldown period for a priority level"""
        return cls.COOLDOWN_PERIODS.get(priority, 10.0)
    
    @classmethod
    def get_movement_threshold(cls, priority: Priority) -> float:
        """Get movement threshold for priority level"""
        return cls.MOVEMENT_THRESHOLDS.get(priority, float('inf'))
    
    @classmethod
    def adjust_priority_by_distance(cls, base_priority: Priority, distance: float) -> float:
        """
        Adjust priority score based on distance
        Returns: Modified priority score (0.0 to 4.0+)
        """
        if base_priority == Priority.IGNORE or base_priority == Priority.LOW:
            return base_priority.value
        
        thresholds = cls.DISTANCE_THRESHOLDS.get(base_priority, {})
        base_score = base_priority.value
        
        if distance < thresholds.get('immediate', 1.0):
            return base_score * 1.5  # Boost priority
        elif distance < thresholds.get('near', 3.0):
            return base_score * 1.0  # Keep priority
        elif distance < thresholds.get('far', 6.0):
            return base_score * 0.7  # Reduce priority
        else:
            return base_score * 0.3  # Significantly reduce priority
    
    @classmethod
    def adjust_priority_by_zone(cls, base_score: float, zone: str) -> float:
        """Adjust priority based on spatial zone"""
        multiplier = cls.SPATIAL_ZONES.get(zone, 1.0)
        return base_score * multiplier
    
    @classmethod
    def detect_context(cls, detected_classes: List[str]) -> List[str]:
        """
        Detect environmental context based on detected objects
        Returns: List of detected contexts
        """
        contexts = []
        detected_set = set(detected_classes)
        
        for context, indicators in cls.CONTEXT_INDICATORS.items():
            # If we find multiple indicators, context is likely
            matches = sum(1 for obj in indicators if obj in detected_set)
            if matches >= 2:  # Need at least 2 indicators
                contexts.append(context)
        
        return contexts


# Announcement templates for different scenarios
ANNOUNCEMENT_TEMPLATES = {
    'immediate_danger': "{object} ahead, {distance} meters!",
    'obstacle_near': "{object} on your {direction}, {distance} meters",
    'moving_object': "{object} approaching from {direction}",
    'scene_summary': "{count} objects detected: {objects}",
    'safe_path': "Path clear ahead",
    'context_change': "Entering {context} area"
}


# Voice settings for different priority levels
VOICE_SETTINGS = {
    Priority.CRITICAL: {
        'volume': 1.0,
        'rate': 1.2,      # Slightly faster
        'pitch': 1.1,     # Slightly higher pitch for urgency
        'interrupt': True  # Can interrupt current speech
    },
    Priority.HIGH: {
        'volume': 0.9,
        'rate': 1.0,
        'pitch': 1.0,
        'interrupt': False
    },
    Priority.MEDIUM: {
        'volume': 0.8,
        'rate': 0.95,
        'pitch': 0.95,
        'interrupt': False
    },
    Priority.LOW: {
        'volume': 0.7,
        'rate': 0.9,
        'pitch': 0.9,
        'interrupt': False
    }
}