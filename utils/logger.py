import logging
from datetime import datetime
from pathlib import Path

class SystemLogger:
    """
    Centralized logging for blind assistance system
    Logs to both file and console
    """
    
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Args:
            log_dir: Directory for log files
            log_level: Minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        
        # Create loggers for different components
        self.detection_logger = self._create_logger("detection", "detection.log")
        self.tracking_logger = self._create_logger("tracking", "tracking.log")
        self.audio_logger = self._create_logger("audio", "audio.log")
        self.system_logger = self._create_logger("system", "system.log")
    
    def _create_logger(self, name, filename):
        """Create logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.log_dir / filename)
        
        # Create formatters
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [{name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Set formatters
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_detection(self, frame_num, num_detections, fps):
        """Log detection results"""
        self.detection_logger.info(f"Frame {frame_num}: {num_detections} objects detected @ {fps:.1f} FPS")
    
    def log_announcement(self, text, priority, obstacle_type):
        """Log audio announcements"""
        self.audio_logger.info(f"ANNOUNCEMENT: \"{text}\" (Priority: {priority}, Type: {obstacle_type})")
    
    def log_tracking(self, num_tracks, new_tracks, lost_tracks):
        """Log tracking status"""
        self.tracking_logger.info(f"{num_tracks} active tracks ({new_tracks} new, {lost_tracks} lost)")
    
    def log_error(self, component, error_msg, exception=None):
        """Log errors with stack trace"""
        logger = getattr(self, f"{component}_logger", self.system_logger)
        logger.error(error_msg, exc_info=exception)
    
    def log_performance(self, metrics):
        """
        Log performance metrics
        
        Args:
            metrics: dict with fps, latency, memory_usage, etc.
        """
        self.system_logger.info(
            f"PERFORMANCE: FPS={metrics.get('fps', 0):.1f}, "
            f"Latency={metrics.get('latency_ms', 0):.1f}ms, "
            f"Memory={metrics.get('memory_mb', 0):.1f}MB, "
            f"CPU={metrics.get('cpu_percent', 0)}%"
        )

