import time
from collections import deque
import numpy as np
import psutil

class PerformanceMonitor:
    """
    Monitors system performance metrics
    Tracks FPS, latency, memory usage, CPU usage
    """
    
    def __init__(self, window_size=100):
        """
        Args:
            window_size: Number of samples for rolling average
        """
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.process = psutil.Process()
    
    def start_frame(self):
        """Mark start of frame processing"""
        return time.time()
    
    def end_frame(self, start_time):
        """
        Mark end of frame processing
        
        Args:
            start_time: Return value from start_frame()
        
        Returns:
            float: Frame processing time in seconds
        """
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        return elapsed
    
    def get_fps(self):
        """Get average FPS over window"""
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def get_latency(self):
        """Get average processing latency in milliseconds"""
        return np.mean(self.frame_times) * 1000 if self.frame_times else 0
    
    def get_memory_usage(self):
        """
        Get current memory usage
        
        Returns:
            dict: {rss: int (bytes), percent: float}
        """
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss,
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def get_stats(self):
        """
        Get all performance statistics
        
        Returns:
            dict: {
                fps: float,
                latency_ms: float,
                memory_mb: float,
                cpu_percent: float,
                uptime_seconds: float
            }
        """
        mem_usage = self.get_memory_usage()
        return {
            'fps': self.get_fps(),
            'latency_ms': self.get_latency(),
            'memory_mb': mem_usage['rss'] / (1024 * 1024),
            'cpu_percent': self.get_cpu_usage(),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def print_stats(self):
        """Print formatted statistics to console"""
        stats = self.get_stats()
        print(
            f"FPS: {stats['fps']:.1f} | "
            f"Latency: {stats['latency_ms']:.1f}ms | "
            f"Memory: {stats['memory_mb']:.1f}MB | "
            f"CPU: {stats['cpu_percent']:.1f}%"
        )
    
    def should_throttle(self):
        """
        Determine if system should throttle to reduce load
        
        Returns:
            bool: True if CPU > 90% or memory > 80%
        """
        mem_usage = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()
        return cpu_usage > 90 or mem_usage['percent'] > 80
