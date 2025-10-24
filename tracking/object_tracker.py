import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.helpers import calculate_iou

class TrackedObject:
    """Represents an object being tracked across frames"""
    
    def __init__(self, detection, track_id, frame_number):
        """
        Args:
            detection: Detection object
            track_id: Unique identifier for this tracked object
            frame_number: Frame where tracking started
        """
        self.track_id = track_id
        self.detections_history = [detection]
        self.positions_history = [detection.center]
        self.first_seen_frame = frame_number
        self.last_seen_frame = frame_number
        self.frames_missing = 0
        self.is_active = True
    
    def update(self, detection, frame_number):
        """Update with new detection"""
        self.detections_history.append(detection)
        self.positions_history.append(detection.center)
        self.last_seen_frame = frame_number
        self.frames_missing = 0
    
    def predict_next_position(self):
        """Predict position in next frame based on velocity"""
        if len(self.positions_history) < 2:
            return self.positions_history[-1]
        
        velocity = self.get_velocity()
        last_position = self.positions_history[-1]
        return (last_position[0] + velocity[0], last_position[1] + velocity[1])
    
    def get_velocity(self):
        """
        Calculate velocity in pixels/frame
        
        Returns:
            (vx, vy): Velocity vector
        """
        if len(self.positions_history) < 2:
            return (0, 0)
        
        # Simple velocity calculation
        last_pos = self.positions_history[-1]
        prev_pos = self.positions_history[-2]
        return (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
    
    def get_age(self):
        """Get number of frames object has been tracked"""
        return self.last_seen_frame - self.first_seen_frame


class ObjectTracker:
    """
    Multi-object tracker using IoU (Intersection over Union) matching
    Simple but effective for slow-moving objects
    """
    
    def __init__(self, max_missing_frames=10, min_iou=0.3):
        """
        Args:
            max_missing_frames: Drop track after this many missed detections
            min_iou: Minimum IoU for matching detection to track
        """
        self.tracked_objects = []
        self.next_track_id = 0
        self.frame_count = 0
        self.max_missing_frames = max_missing_frames
        self.min_iou = min_iou
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects from current frame
        
        Returns:
            List of TrackedObject (active tracks)
        """
        self.frame_count += 1
        
        if not self.tracked_objects:
            for det in detections:
                self._create_new_track(det)
            return self.get_active_tracks()

        matches, unmatched_tracks, unmatched_detections = self.match_detections_to_tracks(detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracked_objects[track_idx].update(detections[det_idx], self.frame_count)

        # Mark unmatched tracks as missing
        for track_idx in unmatched_tracks:
            self.tracked_objects[track_idx].frames_missing += 1
            if self.tracked_objects[track_idx].frames_missing > self.max_missing_frames:
                self.tracked_objects[track_idx].is_active = False

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx])

        # Remove inactive tracks
        self.tracked_objects = [t for t in self.tracked_objects if t.is_active]
        
        return self.get_active_tracks()

    def _create_new_track(self, detection):
        new_track = TrackedObject(detection, self.next_track_id, self.frame_count)
        self.tracked_objects.append(new_track)
        self.next_track_id += 1

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        
        Returns:
            float: IoU value (0 to 1)
        """
        return calculate_iou(bbox1, bbox2)
    
    def match_detections_to_tracks(self, detections):
        """
        Match current detections to existing tracks
        Uses Hungarian algorithm for optimal assignment.
        
        Returns:
            matches: List of (track_idx, detection_idx) tuples
            unmatched_tracks: Indices of tracks with no match
            unmatched_detections: Indices of detections with no match
        """
        if not self.tracked_objects or not detections:
            return [], list(range(len(self.tracked_objects))), list(range(len(detections)))

        iou_matrix = np.zeros((len(self.tracked_objects), len(detections)), dtype=np.float32)

        for t, trk in enumerate(self.tracked_objects):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self.calculate_iou(trk.detections_history[-1].bbox, det.bbox)

        # Use Hungarian algorithm to find optimal assignments
        row_ind, col_ind = linear_sum_assignment(-iou_matrix) # Maximize IoU

        matches = []
        unmatched_tracks = list(range(len(self.tracked_objects)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.min_iou:
                matches.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_detections:
                    unmatched_detections.remove(c)

        return matches, unmatched_tracks, unmatched_detections

    def get_active_tracks(self):
        """Get all currently active tracks"""
        return [t for t in self.tracked_objects if t.is_active]
    
    def get_track_by_id(self, track_id):
        """Get specific track by ID"""
        for track in self.tracked_objects:
            if track.track_id == track_id:
                return track
        return None
