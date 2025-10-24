
"""
Main entry point for Blind Assistance System
Orchestrates all components and runs the main loop
"""

import sys
import time
import argparse
from pathlib import Path

from config.settings import CurrentSettings, SystemMode

# Import all components
from detection.yolo_fastest import YoloFastest
from tracking.object_tracker import ObjectTracker
from tracking.motion_analyzer import MotionAnalyzer
from scene_understanding.spatial_analyzer import SpatialAnalyzer
from scene_understanding.obstacle_classifier import ObstacleClassifier
from scene_understanding.scene_context import SceneContextAnalyzer
from audio.tts_engine import TTSEngine, TTSEngineType
from audio.feedback_manager import FeedbackManager
from audio.announcement_queue import AnnouncementQueue
from input.camera_manager import CameraManager
from utils.logger import SystemLogger
from utils.performance_monitor import PerformanceMonitor


class BlindAssistanceSystem:
    """
    Main system class that coordinates all components
    """
    
    def __init__(self, config=None):
        """
        Initialize the blind assistance system
        
        Args:
            config: Settings object (uses default if None)
        """
        self.config = config or CurrentSettings
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        
        # Initialize logger
        print("📝 Initializing logger...")
        self.logger = SystemLogger(
            log_dir=self.config.LOGS_DIR,
            log_level=self.config.LOGGING_LEVEL
        )
        
        # Initialize performance monitor
        print("📊 Initializing performance monitor...")
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize components
        self._initialize_components()
        
        print("✅ System initialized successfully!\n")
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # 1. Detection
        print("🔍 Loading YOLO detector...")
        self.detector = YoloFastest(
            param_path=str(self.config.MODEL_PARAM_PATH),
            bin_path=str(self.config.MODEL_BIN_PATH),
            target_size=self.config.MODEL_INPUT_SIZE,
            num_threads=self.config.MODEL_NUM_THREADS,
            use_vulkan=self.config.MODEL_USE_VULKAN
        )
        
        if not self.detector.load_model():
            raise RuntimeError("Failed to load detection model!")
        
        # 2. Tracking
        print("🎯 Initializing object tracker...")
        self.tracker = ObjectTracker(
            max_missing_frames=self.config.TRACKING_MAX_MISSING_FRAMES,
            min_iou=self.config.TRACKING_MIN_IOU
        ) if self.config.TRACKING_ENABLED else None
        
        # 3. Motion analysis
        print("🏃 Initializing motion analyzer...")
        self.motion_analyzer = MotionAnalyzer(
            frame_rate=self.config.PERFORMANCE_TARGET_FPS
        ) if self.config.MOTION_DETECTION_ENABLED else None
        
        # 4. Spatial analysis
        print("🗺️  Initializing spatial analyzer...")
        self.spatial_analyzer = SpatialAnalyzer(
            camera_fov=self.config.CAMERA_FOV,
            camera_height=self.config.CAMERA_HEIGHT_METERS,
            frame_width=self.config.CAMERA_WIDTH,
            frame_height=self.config.CAMERA_HEIGHT
        )
        
        # 5. Obstacle classification
        print("⚠️  Initializing obstacle classifier...")
        self.obstacle_classifier = ObstacleClassifier()
        
        # 6. Scene context
        print("🌍 Initializing scene context analyzer...")
        self.context_analyzer = SceneContextAnalyzer()
        
        # 7. Audio system
        if self.config.AUDIO_ENABLED:
            print("🔊 Initializing audio system...")
            
            # TTS engine
            self.tts_engine = TTSEngine(
                engine_type=TTSEngineType(self.config.AUDIO_TTS_ENGINE),
                voice_rate=self.config.AUDIO_VOICE_RATE,
                volume=self.config.AUDIO_VOLUME
            )
            
            # Announcement queue
            self.announcement_queue = AnnouncementQueue(
                max_size=self.config.AUDIO_ANNOUNCEMENT_QUEUE_SIZE
            )
            self.announcement_queue.start_processing(self.tts_engine)
            
            # Feedback manager
            self.feedback_manager = FeedbackManager(
                tts_engine=self.tts_engine,
                announcement_queue=self.announcement_queue,
                max_announcements_per_minute=self.config.AUDIO_MAX_ANNOUNCEMENTS_PER_MINUTE
            )
        else:
            self.tts_engine = None
            self.announcement_queue = None
            self.feedback_manager = None
        
        # 8. Camera
        print("📷 Initializing camera...")
        self.camera = CameraManager(
            camera_id=self.config.CAMERA_ID,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            fps=self.config.CAMERA_FPS
        )
    
    def start(self):
        """Start the system"""
        print("\n" + "="*60)
        print("🚀 STARTING BLIND ASSISTANCE SYSTEM")
        print("="*60)
        
        # Start camera
        if not self.camera.start():
            raise RuntimeError("Failed to start camera!")
        
        self.is_running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            self.logger.log_error("system", "Fatal error in main loop", e)
            raise
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop"""
        
        print("\n🎬 Entering main loop...")
        print("Press Ctrl+C to stop\n")
        
        last_stats_time = time.time()
        
        while self.is_running:
            frame_start = self.perf_monitor.start_frame()
            
            # 1. Capture frame
            frame = self.camera.read_frame(timeout=1.0)
            if frame is None:
                self.logger.log_error("camera", "Failed to read frame")
                continue
            
            # PAUSE CHECK
            if self.is_paused:
                # Still show the camera feed, but do nothing else
                if self.config.VISUALIZATION_ENABLED:
                    vis_frame = self._visualize_frame(frame, [], self.context_analyzer.get_current_context()) # Empty data
                    cv2.imshow('Blind Assistance System', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('p'):
                        self._toggle_pause()
                time.sleep(0.1)
                continue
            
            # 2. Detect objects
            detections = self.detector.detect(
                frame,
                conf_threshold=self.config.MODEL_CONF_THRESHOLD
            )
            
            if self.config.LOGGING_LOG_DETECTIONS:
                self.logger.log_detection(
                    self.frame_count,
                    len(detections),
                    self.perf_monitor.get_fps()
                )
            
            # 3. Track objects (if enabled)
            tracked_objects = None
            if self.tracker and len(detections) > 0:
                tracked_objects = self.tracker.update(detections)
            
            # 4. Spatial analysis
            spatial_infos = []
            if len(detections) > 0:
                spatial_infos = self.spatial_analyzer.analyze(detections)
            
            # 5. Motion analysis (if enabled)
            motion_infos = []
            if self.motion_analyzer and tracked_objects:
                motion_infos = self.motion_analyzer.analyze_motion(
                    tracked_objects
                )
                
                # Update spatial_infos with movement data
                for si in spatial_infos:
                    for mi in motion_infos:
                        if mi.track_id == si.detection.track_id:
                            si.is_moving = mi.is_approaching
                            si.velocity = mi.speed
            
            # 6. Classify obstacles
            obstacles = []
            if len(spatial_infos) > 0:
                obstacles = self.obstacle_classifier.classify(spatial_infos)
            
            # 7. Analyze scene context
            scene_context = self.context_analyzer.analyze(obstacles, self.frame_count)
            
            # 8. Generate and process audio feedback
            if self.feedback_manager:
                self.feedback_manager.process_feedback(obstacles, scene_context)

            
            # 9. Visualization (if enabled)
            if self.config.VISUALIZATION_ENABLED:
                vis_frame = self._visualize_frame(
                    frame,
                    obstacles,
                    scene_context
                )
                
                import cv2
                cv2.imshow('Blind Assistance System', vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    # Pause/resume
                    self._toggle_pause()
                elif key == ord('m'):
                    # Toggle feedback mode
                    self._toggle_mode()            
            # 10. Update performance stats
            frame_time = self.perf_monitor.end_frame(frame_start)
            
            # 11. Periodic stats logging
            if time.time() - last_stats_time > self.config.PERFORMANCE_LOG_INTERVAL:
                if self.config.PERFORMANCE_MONITORING_ENABLED:
                    stats = self.perf_monitor.get_stats()
                    self.logger.log_performance(stats)
                    if self.config.DEBUG_PRINT_PERFORMANCE:
                        self.perf_monitor.print_stats()
                last_stats_time = time.time()
            
            # 12. Throttle if needed
            if self.perf_monitor.should_throttle():
                time.sleep(0.05)  # Brief pause to reduce load
            
            self.frame_count += 1
    
    def _visualize_frame(self, frame, obstacles, scene_context):
        """Add visualizations to frame"""
        import cv2
        vis_frame = frame.copy()
        
        # Draw detections with risk-based colors
        for obs in obstacles:
            det = obs.spatial_info.detection
            x1, y1, x2, y2 = det.bbox
            
            # Color based on risk
            risk_colors = {
                'critical': (0, 0, 255),    # Red
                'high': (0, 165, 255),      # Orange
                'medium': (0, 255, 255),    # Yellow
                'low': (0, 255, 0),         # Green
                'none': (128, 128, 128)     # Gray
            }
            color = risk_colors.get(obs.risk_level.value, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            distance = obs.spatial_info.distance
            dist_str = f"{distance:.1f}m" if distance else obs.spatial_info.distance_zone.value
            
            label = f"{det.class_name} ({dist_str})"
            label += f" - {obs.risk_level.value.upper()}"
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(vis_frame, (x1, y1-30), (x1+label_w+10, y1), color, -1)
            cv2.putText(vis_frame, label, (x1+5, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw info panel
        info = self._get_system_info()
        h, w = frame.shape[:2]
        panel_x = w - 350
        
        # Semi-transparent background
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        
        y = 30
        
        # Title
        cv2.putText(vis_frame, "SYSTEM STATUS", (panel_x+10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        
        # System Info
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(vis_frame, text, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        # Draw FPS on top-left, outside the panel
        if self.config.VISUALIZATION_SHOW_FPS:
            fps_text = f"FPS: {info['FPS']}"
            cv2.putText(vis_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def _get_system_info(self):
        """Return a dictionary of system info for display"""
        perf_stats = self.perf_monitor.get_stats()
        
        info = {
            "FPS": f"{perf_stats['fps']:.1f}",
            "Mode": self.config.FEEDBACK_MODE.value,
            "Paused": self.is_paused,
            "Objects": len(self.obstacle_classifier.get_current_obstacles()),
            "Scene": self.context_analyzer.get_current_context().scene_type.value,
            "Safety": self.context_analyzer.get_current_context().safety_level,
            "Queue": self.announcement_queue.qsize() if self.announcement_queue else 0,
        }
        return info

    def _toggle_pause(self):
        """Toggle pause/resume state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("⏸️ System paused")
            if self.feedback_manager:
                self.feedback_manager.announce_system_state("Paused")
        else:
            print("▶️ System resumed")
            if self.feedback_manager:
                self.feedback_manager.announce_system_state("Resumed")

    def _toggle_mode(self):
        """Toggle feedback mode"""
        # Cycle through modes: quiet -> normal -> verbose -> quiet
        current_mode_index = list(SystemMode).index(self.config.FEEDBACK_MODE)
        next_mode_index = (current_mode_index + 1) % len(SystemMode)
        self.config.FEEDBACK_MODE = list(SystemMode)[next_mode_index]
        
        print(f"🔄 Feedback mode changed to: {self.config.FEEDBACK_MODE.value}")
        if self.feedback_manager:
            self.feedback_manager.announce_system_state(f"Mode {self.config.FEEDBACK_MODE.value}")

    def stop(self):
        """Stop the system and cleanup"""
        print("\n🛑 Stopping system...")
        
        self.is_running = False
        
        # Stop camera
        if self.camera:
            self.camera.stop()
        
        # Stop audio
        if self.announcement_queue:
            self.announcement_queue.stop_processing()
        
        # Cleanup detector
        if self.detector:
            self.detector.cleanup()
        
        # Close windows
        if self.config.VISUALIZATION_ENABLED:
            import cv2
            cv2.destroyAllWindows()
        
        # Save user preferences
        self.config.save_user_preferences()
        
        print("✅ System stopped cleanly")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Blind Assistance System - Real-time obstacle detection and audio feedback"
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['quiet', 'normal', 'verbose'],
        default='normal',
        help='Feedback mode (default: normal)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (headless mode)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    
    parser.add_argument(
        '--pi',
        action='store_true',
        help='Enable Raspberry Pi optimized settings'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set PI_MODE before loading settings
    if args.pi:
        from config import settings
        settings.Settings.PI_MODE = True
        # Reload the settings module to apply the change
        import importlib
        importlib.reload(settings)

    # Apply arguments to settings
    if args.camera is not None:
        CurrentSettings.CAMERA_ID = args.camera
    
    if args.mode:
        CurrentSettings.FEEDBACK_MODE = SystemMode(args.mode)
    
    if args.no_display:
        CurrentSettings.VISUALIZATION_ENABLED = False
    
    if args.debug:
        CurrentSettings.DEBUG_MODE = True
        CurrentSettings.LOGGING_LEVEL = "DEBUG"
    
    # Print settings
    if CurrentSettings.DEBUG_MODE:
        CurrentSettings.print_settings()
    
    # Create and start system
    try:
        system = BlindAssistanceSystem(config=CurrentSettings)
        system.start()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
