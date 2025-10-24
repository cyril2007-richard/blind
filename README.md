# Blind Assistance System

This project is a real-time blind assistance system that uses a camera to detect obstacles and provide audio feedback to the user.

## Project Structure

```
blind-assistance-system/
├── audio/                  # Audio feedback components
│   ├── announcement_queue.py # Manages the queue of announcements
│   ├── feedback_manager.py   # Generates and filters audio feedback
│   ├── sound_effects.py      # (Not yet implemented)
│   └── tts_engine.py         # Text-to-speech engine
├── config/                 # Configuration files
│   ├── priority_rules.py     # Rules for prioritizing obstacles
│   └── settings.py           # System-wide settings
├── detection/              # Object detection components
│   ├── detector_interface.py # Interface for object detectors
│   └── yolo_fastest.py       # YOLO-Fastest object detector
├── input/                  # Input components
│   ├── camera_manager.py     # Manages the camera input
│   └── voice_commands.py     # (Not yet implemented)
├── logs/                   # Log files
├── models/                 # Object detection models
├── scene_understanding/    # Scene analysis components
│   ├── obstacle_classifier.py # Classifies obstacles by risk
│   ├── scene_context.py      # Analyzes the overall scene
│   └── spatial_analyzer.py   # Analyzes the spatial relationship of objects
├── tests/                  # Unit tests
├── tracking/               # Object tracking components
│   ├── motion_analyzer.py    # Analyzes the motion of objects
│   └── object_tracker.py     # Tracks objects across frames
├── utils/                  # Utility scripts
│   ├── helpers.py            # Helper functions
│   ├── logger.py             # Logging utility
│   └── performance_monitor.py # Performance monitoring utility
├── main.py                 # Main entry point of the application
├── requirements.txt        # Python dependencies
├── run.sh                  # Script to run the application
└── README.md               # This file
```

## Getting Started

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**

    ```bash
    ./run.sh
    ```

    By default, the application runs in Raspberry Pi mode. To run in development mode, edit the `run.sh` script and remove the `--pi` flag.
