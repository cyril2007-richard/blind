"""
Obstacle Classifier - Classifies obstacles by importance and risk level
"""

from typing import List, Dict, Tuple
from enum import Enum
import sys
sys.path.append('..')
from config.priority_rules import ObjectPriorityRules, Priority


class RiskLevel(Enum):
    """Risk levels for obstacles"""
    CRITICAL = 4    # Immediate danger
    HIGH = 3           # Significant obstacle
    MEDIUM = 2       # Minor obstacle
    LOW = 1            # Awareness only
    NONE = 0          # No risk


class ObstacleType(Enum):
    """Types of obstacles"""
    MOVING_VEHICLE = "moving_vehicle"
    STATIONARY_VEHICLE = "stationary_vehicle"
    PERSON = "person"
    STRUCTURAL = "structural"  # walls, poles, etc
    GROUND_LEVEL = "ground_level"  # curbs, stairs, etc
    OVERHEAD = "overhead"  # signs, branches, etc
    ANIMAL = "animal"
    UNKNOWN = "unknown"


import itertools

_obstacle_id_counter = itertools.count()

class ObstacleInfo:
    """Complete obstacle information"""
    
    def __init__(self, spatial_info, risk_level: RiskLevel, obstacle_type: ObstacleType,
                 priority_score: float, requires_immediate_action: bool = False):
        self.id = next(_obstacle_id_counter) # Unique ID for each obstacle instance
        self.spatial_info = spatial_info
        self.risk_level = risk_level
        self.obstacle_type = obstacle_type
        self.priority_score = priority_score
        self.requires_immediate_action = requires_immediate_action
        self.avoidance_suggestion = None
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            **self.spatial_info.to_dict(),
            'risk_level': self.risk_level.value,
            'obstacle_type': self.obstacle_type.value,
            'priority_score': self.priority_score,
            'requires_immediate_action': self.requires_immediate_action,
            'avoidance_suggestion': self.avoidance_suggestion
        }
    
    def __repr__(self) -> str:
        return f"Obstacle({self.spatial_info.detection.class_name}, {self.risk_level.value}, score={self.priority_score:.2f})"


class ObstacleClassifier:
    """Classifies and prioritizes obstacles"""
    
    def __init__(self):
        self.priority_rules = ObjectPriorityRules()
        self.current_obstacles = []
        
        # Obstacle type mappings
        self.obstacle_type_map = {
            'car': ObstacleType.MOVING_VEHICLE,
            'truck': ObstacleType.MOVING_VEHICLE,
            'bus': ObstacleType.MOVING_VEHICLE,
            'motorcycle': ObstacleType.MOVING_VEHICLE,
            'bicycle': ObstacleType.MOVING_VEHICLE,
            'person': ObstacleType.PERSON,
            'dog': ObstacleType.ANIMAL,
            'cat': ObstacleType.ANIMAL,
            'fire hydrant': ObstacleType.STRUCTURAL,
            'stop sign': ObstacleType.STRUCTURAL,
            'traffic light': ObstacleType.OVERHEAD,
            'bench': ObstacleType.GROUND_LEVEL,
            'chair': ObstacleType.GROUND_LEVEL,
            'potted plant': ObstacleType.GROUND_LEVEL,
        }
        
    def classify(self, spatial_infos: List) -> List[ObstacleInfo]:
        """
        Classify all spatial objects as obstacles
        
        Args:
            spatial_infos: List of SpatialInfo objects
            
        Returns:
            List of ObstacleInfo objects, sorted by priority
        """
        obstacles = []
        
        for spatial_info in spatial_infos:
            obstacle_info = self._classify_single(spatial_info)
            obstacles.append(obstacle_info)
        
        # Sort by priority score (highest first)
        obstacles.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.current_obstacles = obstacles
        return obstacles
    
    def get_current_obstacles(self) -> List[ObstacleInfo]:
        """Return the latest classified obstacles"""
        return self.current_obstacles
    
    def _classify_single(self, spatial_info) -> ObstacleInfo:
        """Classify a single object"""
        detection = spatial_info.detection
        class_name = detection.class_name
        
        # Get base priority
        base_priority = self.priority_rules.get_priority(class_name)
        
        # Calculate priority score with distance adjustment
        priority_score = self.priority_rules.adjust_priority_by_distance(
            base_priority,
            spatial_info.distance if spatial_info.distance else 3.0  # default 3m if unknown
        )
        
        # Adjust by spatial zone
        zone_name = spatial_info.direction.value.replace(' ', '_')
        priority_score = self.priority_rules.adjust_priority_by_zone(
            priority_score,
            zone_name
        )
        
        # Adjust for movement
        if spatial_info.is_moving:
            priority_score *= 1.5
        
        # Determine risk level
        risk_level = self._determine_risk_level(spatial_info, priority_score)
        
        # Determine obstacle type
        obstacle_type = self.obstacle_type_map.get(class_name, ObstacleType.UNKNOWN)
        
        # Check if immediate action required
        requires_immediate = self._requires_immediate_action(spatial_info, risk_level)
        
        obstacle_info = ObstacleInfo(
            spatial_info=spatial_info,
            risk_level=risk_level,
            obstacle_type=obstacle_type,
            priority_score=priority_score,
            requires_immediate_action=requires_immediate
        )
        
        # Generate avoidance suggestion
        obstacle_info.avoidance_suggestion = self._generate_avoidance_suggestion(obstacle_info)
        
        return obstacle_info
    
    def _determine_risk_level(self, spatial_info, priority_score: float) -> RiskLevel:
        """Determine risk level based on priority score and context"""
        
        # Critical: High priority + close distance
        if priority_score > 5.0 or \
           (spatial_info.distance_zone.value == "immediate" and priority_score > 3.0):
            return RiskLevel.CRITICAL
        
        # High: Medium-high priority or close obstacles
        if priority_score > 3.0 or spatial_info.distance_zone.value == "immediate":
            return RiskLevel.HIGH
        
        # Medium: Some priority or near obstacles
        if priority_score > 1.5 or spatial_info.distance_zone.value == "near":
            return RiskLevel.MEDIUM
        
        # Low: Far or low priority
        if priority_score > 0.5:
            return RiskLevel.LOW
        
        return RiskLevel.NONE
    
    def _requires_immediate_action(self, spatial_info, risk_level: RiskLevel) -> bool:
        """Check if immediate action/announcement is required"""
        
        # Always immediate for critical risk
        if risk_level == RiskLevel.CRITICAL:
            return True
        
        # Immediate if moving object is close
        if spatial_info.is_moving and spatial_info.distance_zone.value in ["immediate", "near"]:
            return True
        
        # Immediate if in center path and close
        if spatial_info.direction.value == "center" and \
           spatial_info.distance_zone.value == "immediate":
            return True
        
        return False
    
    def _generate_avoidance_suggestion(self, obstacle_info: ObstacleInfo) -> str:
        """Generate suggestion for avoiding obstacle"""
        spatial_info = obstacle_info.spatial_info
        direction = spatial_info.direction.value
        
        # For objects in center, suggest direction to avoid
        if direction == "center":
            # Suggest opposite of where most obstacles are
            return "move slightly left or right"
        
        # For objects on sides
        if "left" in direction:
            return "keep to your right"
        if "right" in direction:
            return "keep to your left"
        
        return "proceed with caution"
    
    def get_critical_obstacles(self, obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """Get only critical obstacles"""
        return [obs for obs in obstacles if obs.risk_level == RiskLevel.CRITICAL]
    
    def get_path_blocking_obstacles(self, obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """Get obstacles that are blocking the forward path"""
        return [obs for obs in obstacles 
                if obs.spatial_info.direction.value == "center" 
                and obs.risk_level.value in ["critical", "high"]]