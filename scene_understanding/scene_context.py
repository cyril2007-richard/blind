"""
Scene Context - Understands overall scene and environmental context
"""

from enum import Enum
from typing import List, Dict, Set, Tuple
from collections import Counter
import sys
sys.path.append('..')
from config.priority_rules import ObjectPriorityRules


class SceneType(Enum):
    """Types of scenes/environments"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ROAD_CROSSING = "road_crossing"
    SIDEWALK = "sidewalk"
    PARK = "park"
    BUILDING_ENTRANCE = "building_entrance"
    STAIRS = "stairs"
    UNKNOWN = "unknown"


class SceneContext:
    """Scene context information"""
    
    def __init__(self):
        self.scene_type = SceneType.UNKNOWN
        self.dominant_objects = []
        self.detected_contexts = []
        self.safety_level = "moderate"  # safe, moderate, caution, danger
        self.object_density = 0.0
        self.movement_detected = False
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'scene_type': self.scene_type.value,
            'dominant_objects': self.dominant_objects,
            'detected_contexts': self.detected_contexts,
            'safety_level': self.safety_level,
            'object_density': self.object_density,
            'movement_detected': self.movement_detected
        }


class SceneContextAnalyzer:
    """Analyzes overall scene context"""
    
    def __init__(self):
        self.priority_rules = ObjectPriorityRules()
        self.context_history = []  # Track context over time
        self.max_history = 10
        self.current_context = SceneContext()
        
    def analyze(self, obstacles: List, frame_count: int = 0) -> SceneContext:
        """
        Analyze scene context from obstacles
        
        Args:
            obstacles: List of ObstacleInfo objects
            frame_count: Current frame number for temporal analysis
            
        Returns:
            SceneContext object
        """
        context = SceneContext()
        
        if not obstacles:
            return context
        
        # Extract detected class names
        detected_classes = [obs.spatial_info.detection.class_name for obs in obstacles]
        
        # Detect contexts using priority rules
        context.detected_contexts = self.priority_rules.detect_context(detected_classes)
        
        # Determine scene type
        context.scene_type = self._determine_scene_type(detected_classes, context.detected_contexts)
        
        # Find dominant objects (most common)
        class_counter = Counter(detected_classes)
        context.dominant_objects = [cls for cls, _ in class_counter.most_common(3)]
        
        # Calculate object density
        context.object_density = len(obstacles) / 10.0  # Normalize to 0-1+ range
        
        # Check for movement
        context.movement_detected = any(obs.spatial_info.is_moving for obs in obstacles)
        
        # Determine safety level
        context.safety_level = self._assess_safety(obstacles, context)
        
        # Update history
        self._update_history(context)
        
        self.current_context = context
        return context
    
    def get_current_context(self) -> SceneContext:
        """Return the latest analyzed scene context"""
        return self.current_context
    
    def _determine_scene_type(self, detected_classes: List[str], 
                              detected_contexts: List[str]) -> SceneType:
        """Determine the type of scene"""
        
        # Check specific contexts first
        if 'road_crossing' in detected_contexts:
            return SceneType.ROAD_CROSSING
        
        if 'stairs_area' in detected_contexts:
            return SceneType.STAIRS
        
        if 'indoor' in detected_contexts:
            return SceneType.INDOOR
        
        # Check for outdoor indicators
        outdoor_indicators = ['car', 'truck', 'bus', 'bicycle', 'traffic light', 'stop sign']
        outdoor_count = sum(1 for obj in detected_classes if obj in outdoor_indicators)
        
        if outdoor_count >= 2:
            # Distinguish between sidewalk and road crossing
            if any(obj in ['traffic light', 'stop sign'] for obj in detected_classes):
                return SceneType.ROAD_CROSSING
            return SceneType.SIDEWALK
        
        # Check for park indicators
        park_indicators = ['bench', 'dog', 'person', 'bird']
        park_count = sum(1 for obj in detected_classes if obj in park_indicators)
        if park_count >= 2:
            return SceneType.PARK
        
        # Check for building entrance
        if 'door' in detected_classes or 'stairs' in detected_classes:
            return SceneType.BUILDING_ENTRANCE
        
        # Default
        return SceneType.OUTDOOR if outdoor_count > 0 else SceneType.UNKNOWN
    
    def _assess_safety(self, obstacles: List, context: SceneContext) -> str:
        """Assess overall safety level of the scene"""
        
        # Count obstacles by risk level
        critical_count = sum(1 for obs in obstacles if obs.risk_level.value == "critical")
        high_count = sum(1 for obs in obstacles if obs.risk_level.value == "high")
        
        # Danger: Multiple critical obstacles or moving vehicles nearby
        if critical_count >= 2 or \
           (critical_count >= 1 and context.movement_detected):
            return "danger"
        
        # Caution: Critical obstacles or multiple high-risk obstacles
        if critical_count >= 1 or high_count >= 3:
            return "caution"
        
        # Safe: Few or no significant obstacles
        if high_count == 0 and context.object_density < 0.3:
            return "safe"
        
        # Default: Moderate
        return "moderate"
    
    def _update_history(self, context: SceneContext):
        """Update context history for temporal analysis"""
        self.context_history.append(context)
        
        # Keep only recent history
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
    
    def get_scene_summary(self, context: SceneContext) -> str:
        """Generate a human-readable scene summary"""
        
        summary_parts = []
        
        # Scene type
        summary_parts.append(f"Environment: {context.scene_type.value}")
        
        # Safety level
        if context.safety_level != "moderate":
            summary_parts.append(f"Safety: {context.safety_level}")
        
        # Dominant objects
        if context.dominant_objects:
            objects_str = ", ".join(context.dominant_objects[:2])
            summary_parts.append(f"Nearby: {objects_str}")
        
        # Object density
        if context.object_density > 0.7:
            summary_parts.append("crowded area")
        elif context.object_density < 0.2:
            summary_parts.append("clear area")
        
        return ". ".join(summary_parts)
    
    def detect_scene_change(self) -> bool:
        """Detect if scene has significantly changed"""
        if len(self.context_history) < 3:
            return False
        
        # Compare recent contexts
        recent = self.context_history[-3:]
        scene_types = [ctx.scene_type for ctx in recent]
        
        # Scene changed if types are different
        return len(set(scene_types)) > 1
    
    def is_scene_stable(self) -> bool:
        """Check if scene has been stable over recent history"""
        if len(self.context_history) < 5:
            return False
        
        recent = self.context_history[-5:]
        
        # Check if scene type is consistent
        scene_types = [ctx.scene_type for ctx in recent]
        return len(set(scene_types)) == 1