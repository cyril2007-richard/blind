"""
System-wide configuration settings
All tunable parameters in one place
"""

from pathlib import Path
from enum import Enum


class SystemMode(Enum):
    """Operating modes"""
    NORMAL = "normal"      # Standard announcements
    QUIET = "quiet"        # Only critical announcements
    VERBOSE = "verbose"    # All announcements including low priority
    SILENT = "silent"      # No announcements (testing mode)


class Settings:
    """System configuration settings"""
    
    # ==================== DEPLOYMENT SETTINGS ====================
    # Set to True when deploying on a Raspberry Pi for optimized performance
    PI_MODE = False
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    CONFIG_FILE = BASE_DIR / "user_config.json"
    
    # ==================== MODEL SETTINGS ====================
    MODEL_PARAM_PATH = MODELS_DIR / "yolo-fastest-1.1.param"
    MODEL_BIN_PATH = MODELS_DIR / "yolo-fastest-1.1.bin"
    MODEL_INPUT_SIZE = 320
    MODEL_CONF_THRESHOLD = 0.4
    MODEL_NUM_THREADS = 4
    MODEL_USE_VULKAN = True  # GPU acceleration
    
    # ==================== CAMERA SETTINGS ====================
    CAMERA_ID = 0  # Default camera (0 for USB, or video file path)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    CAMERA_FOV = 62.0  # Field of view in degrees (Pi Camera V2)
    CAMERA_HEIGHT_METERS = 1.5  # Camera mounting height
    CAMERA_WARMUP_FRAMES = 10  # Skip first N frames
    CAMERA_RECONNECT_ATTEMPTS = 3
    CAMERA_RECONNECT_DELAY = 2.0  # Seconds between reconnection attempts
    
    # ==================== DETECTION SETTINGS ====================
    DETECTION_MIN_CONFIDENCE = 0.4
    DETECTION_MAX_OBJECTS = 20  # Max objects to process per frame
    DETECTION_SKIP_FRAMES = 0  # Skip N frames (0 = process all)
    
    # ==================== TRACKING SETTINGS ====================
    TRACKING_ENABLED = True
    TRACKING_MAX_MISSING_FRAMES = 10  # Drop track after N missed frames
    TRACKING_MIN_IOU = 0.3  # Minimum IoU for matching
    TRACKING_HISTORY_SIZE = 10  # Number of positions to keep
    TRACKING_MIN_TRACK_LENGTH = 3  # Minimum frames before considering valid
    
    # ==================== MOTION DETECTION SETTINGS ====================
    MOTION_DETECTION_ENABLED = True
    MOTION_VELOCITY_THRESHOLD = 2.0  # Pixels/frame to consider moving
    MOTION_APPROACHING_THRESHOLD = 1.05  # Bbox size increase ratio
    MOTION_TTC_WARNING_SECONDS = 5.0  # Warn if time-to-collision < N seconds
    
    # ==================== SPATIAL ANALYSIS SETTINGS ====================
    SPATIAL_IMMEDIATE_DISTANCE = 2.0  # Meters
    SPATIAL_NEAR_DISTANCE = 5.0  # Meters
    SPATIAL_FAR_DISTANCE = 10.0  # Meters
    
    # Distance zones as tuples (min, max) in meters
    DISTANCE_ZONES = {
        'immediate': (0, 2.0),
        'near': (2.0, 5.0),
        'far': (5.0, float('inf'))
    }
    
    # ==================== AUDIO SETTINGS ====================
    AUDIO_ENABLED = True
    AUDIO_TTS_ENGINE = "pyttsx3"  # 'pyttsx3' (offline) or 'gtts' (online)
    AUDIO_VOICE_RATE = 150  # Words per minute
    AUDIO_VOLUME = 0.9  # 0.0 to 1.0
    AUDIO_PITCH = 1.0  # Voice pitch multiplier
    
    # Rate limiting
    AUDIO_MAX_ANNOUNCEMENTS_PER_MINUTE = 10
    AUDIO_ANNOUNCEMENT_QUEUE_SIZE = 50
    
    # Voice properties per priority (can override for urgent messages)
    AUDIO_CRITICAL_RATE = 180  # Faster speech for critical
    AUDIO_CRITICAL_VOLUME = 1.0  # Max volume for critical
    
    # Cooldown periods (seconds) - how long before re-announcing same object
    AUDIO_GLOBAL_COOLDOWN = 1.5 # Minimum time between any two announcements
    AUDIO_COOLDOWN_CRITICAL = 3.0
    AUDIO_COOLDOWN_HIGH = 5.0
    AUDIO_COOLDOWN_MEDIUM = 10.0
    AUDIO_COOLDOWN_LOW = 30.0
    
    # Scene summary settings
    AUDIO_SCENE_SUMMARY_INTERVAL = 15.0  # Seconds between summaries
    AUDIO_SCENE_SUMMARY_ENABLED = True
    
    # Sound effects
    AUDIO_USE_SOUND_EFFECTS = True
    AUDIO_BEEP_ENABLED = True
    
    # ==================== FEEDBACK SETTINGS ====================
    FEEDBACK_MODE = SystemMode.NORMAL  # Default mode
    
    # What to announce in each mode
    FEEDBACK_NORMAL_MIN_PRIORITY = 2  # Announce MEDIUM and above
    FEEDBACK_QUIET_MIN_PRIORITY = 3   # Announce HIGH and above
    FEEDBACK_VERBOSE_MIN_PRIORITY = 1 # Announce LOW and above
    
    # ==================== PERFORMANCE SETTINGS ====================
    PERFORMANCE_TARGET_FPS = 15  # Target processing FPS
    PERFORMANCE_MIN_FPS = 10  # Below this, start throttling
    PERFORMANCE_MONITORING_ENABLED = True
    PERFORMANCE_LOG_INTERVAL = 30  # Log stats every N seconds
    
    # Memory limits
    PERFORMANCE_MAX_MEMORY_MB = 800  # Alert if exceeding (Pi 3B+ has 1GB)
    PERFORMANCE_MAX_CPU_PERCENT = 90  # Alert if exceeding
    
    # ==================== LOGGING SETTINGS ====================
    LOGGING_ENABLED = True
    LOGGING_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOGGING_TO_FILE = True
    LOGGING_TO_CONSOLE = True
    LOGGING_MAX_FILE_SIZE_MB = 10
    LOGGING_BACKUP_COUNT = 5  # Keep last N log files
    
    # What to log
    LOGGING_LOG_DETECTIONS = True
    LOGGING_LOG_ANNOUNCEMENTS = True
    LOGGING_LOG_TRACKING = True
    LOGGING_LOG_PERFORMANCE = True
    
    # ==================== VOICE COMMANDS SETTINGS ====================
    VOICE_COMMANDS_ENABLED = False  # Disabled by default (requires extra resources)
    VOICE_COMMANDS_WAKE_WORD = "hey assistant"
    VOICE_COMMANDS_LANGUAGE = "en-US"
    VOICE_COMMANDS_TIMEOUT = 5.0  # Seconds to wait for command after wake word
    
    # ==================== UI/VISUALIZATION SETTINGS ====================
    VISUALIZATION_ENABLED = True  # Show video feed with detections
    VISUALIZATION_SHOW_BBOXES = True
    VISUALIZATION_SHOW_LABELS = True
    VISUALIZATION_SHOW_INFO_PANEL = True
    VISUALIZATION_SHOW_FPS = True
    VISUALIZATION_FONT_SCALE = 0.6
    VISUALIZATION_LINE_THICKNESS = 2
    
    # ==================== SAFETY SETTINGS ====================
    SAFETY_EMERGENCY_STOP_ENABLED = True
    SAFETY_MAX_CRITICAL_OBJECTS = 5  # If more, suggest stopping
    SAFETY_PATH_WIDTH_METERS = 1.0  # Width of forward path to check
    
    # ==================== CALIBRATION SETTINGS ====================
    # These can be calibrated per device/setup
    CALIBRATION_PIXEL_TO_METER_RATIO = None  # Auto-calculate if None
    CALIBRATION_KNOWN_OBJECT_HEIGHTS = {
        # Real-world heights in meters for distance estimation
        'person': 1.7,
        'car': 1.5,
        'truck': 2.5,
        'bus': 3.0,
        'bicycle': 1.1,
        'motorcycle': 1.2,
        'dog': 0.6,
        'cat': 0.3,
        'fire hydrant': 0.7,
        'stop sign': 2.5,
        'traffic light': 3.5,
        'bench': 0.5,
        'chair': 0.9
    }
    
    # ==================== DEBUG SETTINGS ====================
    DEBUG_MODE = False
    DEBUG_SAVE_FRAMES = False  # Save processed frames to disk
    DEBUG_SAVE_INTERVAL = 30  # Save every Nth frame
    DEBUG_SAVE_DIR = BASE_DIR / "debug_frames"
    DEBUG_PRINT_DETECTIONS = False
    DEBUG_PRINT_ANNOUNCEMENTS = True
    DEBUG_PRINT_PERFORMANCE = False
    
    # ==================== METHODS ====================
    
    @classmethod
    def load_user_preferences(cls):
        """
        Load user preferences from config file
        Overrides default settings with user's saved preferences
        """
        if cls.CONFIG_FILE.exists():
            try:
                import json
                with open(cls.CONFIG_FILE, 'r') as f:
                    prefs = json.load(f)
                
                # Update settings from preferences
                for key, value in prefs.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                
                print(f"✓ Loaded user preferences from {cls.CONFIG_FILE}")
            except Exception as e:
                print(f"⚠ Warning: Could not load preferences: {e}")
    
    @classmethod
    def save_user_preferences(cls):
        """Save current settings as user preferences"""
        try:
            import json
            
            # Collect all settings
            prefs = {
                'FEEDBACK_MODE': cls.FEEDBACK_MODE.value,
                'AUDIO_VOLUME': cls.AUDIO_VOLUME,
                'AUDIO_VOICE_RATE': cls.AUDIO_VOICE_RATE,
                'AUDIO_MAX_ANNOUNCEMENTS_PER_MINUTE': cls.AUDIO_MAX_ANNOUNCEMENTS_PER_MINUTE,
                'CAMERA_ID': cls.CAMERA_ID,
                'MODEL_CONF_THRESHOLD': cls.MODEL_CONF_THRESHOLD,
                'VISUALIZATION_ENABLED': cls.VISUALIZATION_ENABLED,
                'VOICE_COMMANDS_ENABLED': cls.VOICE_COMMANDS_ENABLED,
            }
            
            cls.CONFIG_FILE.parent.mkdir(exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(prefs, indent=2, fp=f)
            
            print(f"✓ Saved preferences to {cls.CONFIG_FILE}")
        except Exception as e:
            print(f"⚠ Warning: Could not save preferences: {e}")
    
    @classmethod
    def reset_to_defaults(cls):
        """Reset all settings to defaults"""
        # Could reload this file or manually reset values
        pass
    
    @classmethod
    def validate_settings(cls):
        """
        Validate settings and warn about issues
        Returns: List of warning messages
        """
        warnings = []
        
        # Check paths exist
        if not cls.MODELS_DIR.exists():
            warnings.append(f"Models directory not found: {cls.MODELS_DIR}")
        
        if not cls.MODEL_PARAM_PATH.exists():
            warnings.append(f"Model param file not found: {cls.MODEL_PARAM_PATH}")
        
        if not cls.MODEL_BIN_PATH.exists():
            warnings.append(f"Model bin file not found: {cls.MODEL_BIN_PATH}")
        
        # Check value ranges
        if not 0 <= cls.AUDIO_VOLUME <= 1:
            warnings.append(f"Audio volume out of range: {cls.AUDIO_VOLUME}")
        
        if cls.CAMERA_WIDTH <= 0 or cls.CAMERA_HEIGHT <= 0:
            warnings.append(f"Invalid camera resolution: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        
        return warnings
    
    @classmethod
    def print_settings(cls):
        """Print all current settings"""
        print("\n" + "="*60)
        print("SYSTEM SETTINGS")
        print("="*60)
        
        sections = {
            'Model': ['MODEL_PARAM_PATH', 'MODEL_BIN_PATH', 'MODEL_INPUT_SIZE', 'MODEL_CONF_THRESHOLD'],
            'Camera': ['CAMERA_ID', 'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'CAMERA_FPS'],
            'Audio': ['AUDIO_TTS_ENGINE', 'AUDIO_VOLUME', 'AUDIO_VOICE_RATE'],
            'Feedback': ['FEEDBACK_MODE', 'AUDIO_MAX_ANNOUNCEMENTS_PER_MINUTE'],
            'Performance': ['PERFORMANCE_TARGET_FPS', 'TRACKING_ENABLED', 'MOTION_DETECTION_ENABLED'],
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = getattr(cls, key, 'N/A')
                print(f"  {key}: {value}")
        
        print("="*60 + "\n")


class PiSettings(Settings):
    """Optimized settings for Raspberry Pi"""
    
    # ==================== MODEL SETTINGS ====================
    MODEL_NUM_THREADS = 2  # Use fewer threads on Pi
    MODEL_USE_VULKAN = False # Vulkan may not be available/stable
    
    # ==================== CAMERA SETTINGS ====================
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    CAMERA_FPS = 15
    
    # ==================== PERFORMANCE SETTINGS ====================
    PERFORMANCE_TARGET_FPS = 10
    
    # ==================== LOGGING SETTINGS ====================
    LOGGING_ENABLED = False
    LOGGING_TO_CONSOLE = False
    
    # ==================== UI/VISUALIZATION SETTINGS ====================
    VISUALIZATION_ENABLED = False


# Load user preferences on module import
# Choose settings class based on PI_MODE
if Settings.PI_MODE:
    print("🚀 Using Raspberry Pi optimized settings")
    CurrentSettings = PiSettings
else:
    CurrentSettings = Settings

CurrentSettings.load_user_preferences()

# Validate settings and print warnings
warnings = CurrentSettings.validate_settings()
if warnings:
    print("\n⚠ Configuration Warnings:")
    for warning in warnings:
        print(f"  - {warning}")
