from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .object_tracker import TrackedObject

@dataclass
class MotionAnalysis:
    track_id: int
    speed: float  # in pixels/frame
    direction_vector: Tuple[float, float]
    is_approaching: bool
    time_to_collision: float  # in frames, infinity if not approaching

class MotionAnalyzer:
    """Analyzes the motion of tracked objects."""

    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate

    def analyze_motion(self, tracked_objects: List[TrackedObject]) -> List[MotionAnalysis]:
        """
        Analyze the motion of a list of tracked objects.

        Args:
            tracked_objects: A list of TrackedObject instances.

        Returns:
            A list of MotionAnalysis objects.
        """
        motion_analyses = []
        for obj in tracked_objects:
            if len(obj.positions_history) < 2:
                continue

            speed, direction_vector = self._calculate_speed_and_direction(obj)
            is_approaching, time_to_collision = self._estimate_time_to_collision(obj)

            motion_analyses.append(
                MotionAnalysis(
                    track_id=obj.track_id,
                    speed=speed,
                    direction_vector=direction_vector,
                    is_approaching=is_approaching,
                    time_to_collision=time_to_collision,
                )
            )
        return motion_analyses

    def _calculate_speed_and_direction(self, obj: TrackedObject) -> Tuple[float, Tuple[float, float]]:
        """Calculates the speed and direction of an object."""
        vx, vy = obj.get_velocity()
        speed = np.sqrt(vx**2 + vy**2)
        direction_vector = (vx, vy)
        return speed, direction_vector

    def _estimate_time_to_collision(self, obj: TrackedObject) -> Tuple[bool, float]:
        """
        Estimates the time to collision with the bottom of the frame.
        Assumes the camera is mounted on the user.
        """
        if len(obj.positions_history) < 2:
            return False, float('inf')

        last_pos = obj.positions_history[-1]
        prev_pos = obj.positions_history[-2]

        # Check if the object is moving downwards (approaching)
        is_approaching = last_pos[1] > prev_pos[1]

        if not is_approaching:
            return False, float('inf')

        # Simple time to collision estimation
        # This is a very rough estimate and can be improved
        velocity_y = last_pos[1] - prev_pos[1]
        if velocity_y <= 0:
            return False, float('inf')

        # Assuming the "collision zone" is the bottom of the frame
        # This needs to be calibrated based on the camera setup
        collision_y = 480  # Example: frame height
        distance_to_collision = collision_y - last_pos[1]
        time_to_collision = distance_to_collision / velocity_y

        return True, time_to_collision
