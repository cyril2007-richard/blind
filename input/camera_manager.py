import cv2

class CameraManager:
    """
    Manages camera capture with error handling and optimization
    """
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Args:
            camera_id: Camera device ID (0 for default, or video file path)
            width: Frame width
            height: Frame height
            fps: Target frame rate
        """
        self.camera_id = camera_id
        self.capture = None
    
    def start(self):
        """
        Start camera capture
        
        Returns:
            bool: True if successful
        """
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        return True
    
    def stop(self):
        """Stop camera capture and release resources"""
        if self.capture:
            self.capture.release()
        print("Camera stopped.")
    
    def read_frame(self, timeout=1.0):
        """
        Read latest frame from buffer
        
        Args:
            timeout: Max seconds to wait for frame
        
        Returns:
            numpy array: Frame (BGR format) or None if failed
        """
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame
    
    def get_camera_info(self):
        """
        Get camera properties
        
        Returns:
            dict: {width, height, fps, backend, format}
        """
        if not self.capture:
            return {}
        return {
            'width': self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.capture.get(cv2.CAP_PROP_FPS),
            'backend': self.capture.getBackendName(),
            'format': self.capture.get(cv2.CAP_PROP_FORMAT)
        }
    
    def is_camera_available(self):
        """Check if camera is working"""
        return self.capture is not None and self.capture.isOpened()