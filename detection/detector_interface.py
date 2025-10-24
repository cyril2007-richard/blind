"""
Abstract interface for object detectors
Allows easy swapping of different detection models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class Detection:
    """Standardized detection result"""
    
    def __init__(self, bbox: List[int], confidence: float, class_id: int, class_name: str):
        """
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence score (0-1)
            class_id: Integer class ID
            class_name: Human-readable class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.timestamp = None  # Will be set by tracker
        self.track_id = None   # Will be set by tracker
        
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> int:
        """Get width of bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area,
            'track_id': self.track_id,
            'timestamp': self.timestamp
        }
    
    def __repr__(self) -> str:
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class DetectorInterface(ABC):
    """Abstract base class for object detectors"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize detector
        
        Args:
            model_path: Path to model files
            **kwargs: Additional detector-specific parameters
        """
        self.model_path = model_path
        self.class_names = []
        self.is_initialized = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the detection model
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """
        Run detection on image
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model metadata (name, version, input_size, etc.)
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing step (can be overridden)
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return image
    
    def postprocess(self, detections: List[Detection]) -> List[Detection]:
        """
        Optional postprocessing step (can be overridden)
        Apply NMS, filtering, etc.
        
        Args:
            detections: Raw detections
            
        Returns:
            Processed detections
        """
        return detections
    
    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detections to draw
            
        Returns:
            Image with drawn detections
        """
        import cv2
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            color = self._get_color(det.class_id)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img_copy, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_copy
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources (can be overridden)"""
        pass


class DetectorFactory:
    """Factory for creating detector instances"""
    
    _detectors = {}
    
    @classmethod
    def register(cls, name: str, detector_class: type):
        """Register a detector class"""
        cls._detectors[name] = detector_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> DetectorInterface:
        """
        Create a detector instance
        
        Args:
            name: Detector name (e.g., 'yolo_fastest')
            **kwargs: Arguments to pass to detector constructor
            
        Returns:
            Detector instance
        """
        if name not in cls._detectors:
            raise ValueError(f"Unknown detector: {name}. Available: {list(cls._detectors.keys())}")
        
        detector_class = cls._detectors[name]
        return detector_class(**kwargs)
    
    @classmethod
    def list_detectors(cls) -> List[str]:
        """List all registered detectors"""
        return list(cls._detectors.keys())