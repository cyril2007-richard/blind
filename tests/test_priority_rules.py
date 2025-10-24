"""
Tests for priority rules
"""

import sys
sys.path.append('..')
from config.priority_rules import ObjectPriorityRules, Priority


def test_get_priority():
    """Test basic priority retrieval"""
    # Critical objects
    assert ObjectPriorityRules.get_priority("car") == Priority.CRITICAL
    assert ObjectPriorityRules.get_priority("truck") == Priority.CRITICAL
    
    # High priority
    assert ObjectPriorityRules.get_priority("person") == Priority.HIGH
    
    # Medium priority
    assert ObjectPriorityRules.get_priority("chair") == Priority.MEDIUM
    
    # Low priority
    assert ObjectPriorityRules.get_priority("book") == Priority.LOW
    
    # Unknown object defaults to LOW
    assert ObjectPriorityRules.get_priority("unknown_object") == Priority.LOW


def test_should_track_movement():
    """Test movement tracking flags"""
    assert ObjectPriorityRules.should_track_movement("car") == True
    assert ObjectPriorityRules.should_track_movement("person") == True
    assert ObjectPriorityRules.should_track_movement("book") == False


def test_adjust_priority_by_distance():
    """Test distance-based priority adjustment"""
    base_priority = Priority.CRITICAL
    
    # Very close - should boost priority
    close_score = ObjectPriorityRules.adjust_priority_by_distance(base_priority, 1.0)
    assert close_score > base_priority.value
    
    # Far - should reduce priority
    far_score = ObjectPriorityRules.adjust_priority_by_distance(base_priority, 15.0)
    assert far_score < base_priority.value
    
    print(f"✓ Priority adjustment works: close={close_score:.2f}, far={far_score:.2f}")


def test_detect_context():
    """Test context detection"""
    # Road crossing context
    road_objects = ['car', 'truck', 'traffic light', 'stop sign']
    contexts = ObjectPriorityRules.detect_context(road_objects)
    assert 'road_crossing' in contexts
    print(f"✓ Detected contexts: {contexts}")
    
    # Indoor context
    indoor_objects = ['chair', 'couch', 'tv', 'dining table']
    contexts = ObjectPriorityRules.detect_context(indoor_objects)
    assert 'indoor' in contexts
    print(f"✓ Detected indoor context")


def test_cooldown_periods():
    """Test cooldown retrieval"""
    critical_cooldown = ObjectPriorityRules.get_cooldown(Priority.CRITICAL)
    low_cooldown = ObjectPriorityRules.get_cooldown(Priority.LOW)
    
    assert critical_cooldown < low_cooldown
    print(f"✓ Cooldowns: CRITICAL={critical_cooldown}s, LOW={low_cooldown}s")


if __name__ == "__main__":
    print("Testing Priority Rules...\n")
    
    test_get_priority()
    print("✓ Priority retrieval works")
    
    test_should_track_movement()
    print("✓ Movement tracking flags work")
    
    test_adjust_priority_by_distance()
    test_detect_context()
    test_cooldown_periods()
    
    print("\n✅ All priority rules tests passed!")