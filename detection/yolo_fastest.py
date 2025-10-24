"""
YOLO-Fastest detector implementation
Optimized for real-time detection on Raspberry Pi
Implements DetectorInterface for standardization
"""

import cv2
import ncnn
import numpy as np
from typing import List, Dict, Any
from .detector_interface import DetectorInterface, Detection, DetectorFactory


class YoloFastest(DetectorInterface):
    """YOLO-Fastest detector with NCNN backend"""
    
    def __init__(self, param_path: str, bin_path: str, target_size: int = 320, 
                 num_threads: int = 4, use_vulkan: bool = True):
        """
        Initialize YOLO-Fastest detector
        
        Args:
            param_path: Path to .param file
            bin_path: Path to .bin file
            target_size: Input size for model (default 320)
            num_threads: Number of CPU threads
            use_vulkan: Use Vulkan GPU acceleration
        """
        super().__init__(param_path)
        
        self.param_path = param_path
        self.bin_path = bin_path
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_vulkan = use_vulkan
        
        # Preprocessing parameters
        self.mean_vals = [0, 0, 0]
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]
        
        # COCO class names
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # Pre-generate colors for visualization
        self.class_colors = {}
        for i in range(len(self.class_names)):
            np.random.seed(i)
            self.class_colors[i] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        self.net = None
        self.debug_printed = False
        
    def load_model(self) -> bool:
        """Load NCNN model"""
        try:
            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = self.use_vulkan
            self.net.opt.num_threads = self.num_threads
            
            # Load model files
            ret_param = self.net.load_param(self.param_path)
            ret_model = self.net.load_model(self.bin_path)
            
            if ret_param != 0 or ret_model != 0:
                print(f"Error loading model: param={ret_param}, model={ret_model}")
                return False
            
            self.is_initialized = True
            print(f"✓ YOLO-Fastest loaded successfully")
            print(f"  - Input size: {self.target_size}x{self.target_size}")
            print(f"  - Threads: {self.num_threads}")
            print(f"  - Vulkan: {self.use_vulkan}")
            
            return True
            
        except Exception as e:
            print(f"Error loading YOLO-Fastest: {e}")
            return False
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[Detection]:
        """
        Run detection on image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized or self.net is None:
            print("Model not initialized!")
            return []
        
        img_h, img_w = image.shape[:2]
        
        # Preprocessing
        mat_in = ncnn.Mat.from_pixels_resize(
            image,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img_w,
            img_h,
            self.target_size,
            self.target_size
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        
        # Forward pass
        ex = self.net.create_extractor()
        ex.input("data", mat_in)
        ret, mat_out = ex.extract("output")
        
        if ret != 0:
            print(f"Extraction failed with code: {ret}")
            return []
        
        # Debug output format once
        if not self.debug_printed:
            print(f"\n=== YOLO-Fastest Output ===")
            print(f"Dimensions: w={mat_out.w}, h={mat_out.h}, c={mat_out.c}")
            if mat_out.h > 0:
                sample = mat_out.row(0)
                print(f"Sample row: {len(sample)} values")
                print(f"Sample: {sample[:min(10, len(sample))]}")
            print(f"==========================\n")
            self.debug_printed = True
        
        # Parse detections
        detections = []
        num_classes = len(self.class_names)
        
        for i in range(mat_out.h):
            values = mat_out.row(i)
            
            # Validate row format
            if len(values) < 6:
                continue
            
            class_id = int(values[0]) - 1  # YOLO uses 1-indexed classes
            confidence = values[1]
            
            # Filter by confidence and valid class
            if confidence <= conf_threshold or class_id < 0 or class_id >= num_classes:
                continue
            
            # Convert normalized coords to pixels
            x1 = max(0, min(int(values[2] * img_w), img_w - 1))
            y1 = max(0, min(int(values[3] * img_h), img_h - 1))
            x2 = max(0, min(int(values[4] * img_w), img_w - 1))
            y2 = max(0, min(int(values[5] * img_h), img_h - 1))
            
            # Validate box
            if x2 > x1 and y2 > y1:
                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=self.class_names[class_id]
                )
                detections.append(detection)
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': 'YOLO-Fastest',
            'version': '1.0',
            'backend': 'NCNN',
            'input_size': self.target_size,
            'num_classes': len(self.class_names),
            'threads': self.num_threads,
            'vulkan': self.use_vulkan,
            'param_path': self.param_path,
            'bin_path': self.bin_path
        }
    
    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detections on image with optimized rendering
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with visualizations
        """
        img_copy = image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.class_colors.get(det.class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background
            y_label = max(y1 - 10, label_h + 10)
            cv2.rectangle(
                img_copy,
                (x1, y_label - label_h - 10),
                (x1 + label_w + 10, y_label),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_copy,
                label,
                (x1 + 5, y_label - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return img_copy
    
    def cleanup(self):
        """Cleanup NCNN resources"""
        if self.net is not None:
            self.net.clear()
            self.net = None
        self.is_initialized = False


# Register with factory
DetectorFactory.register('yolo_fastest', YoloFastest)