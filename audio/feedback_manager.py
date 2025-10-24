import time
from collections import deque

from config.priority_rules import ObjectPriorityRules, ANNOUNCEMENT_TEMPLATES, VOICE_SETTINGS, Priority
from config.settings import Settings, SystemMode
from scene_understanding.scene_context import SceneContext
from scene_understanding.obstacle_classifier import ObstacleInfo

class FeedbackManager:
    """
    Manages all audio feedback to the user
    Implements smart filtering to avoid audio overload
    """
    
    def __init__(self, tts_engine, announcement_queue, max_announcements_per_minute=10):
        """
        Args:
            tts_engine: Instance of TTSEngine
            announcement_queue: Instance of AnnouncementQueue
            max_announcements_per_minute: Rate limit for announcements
        """
        self.tts_engine = tts_engine
        self.announcement_queue = announcement_queue
        self.max_announcements_per_minute = max_announcements_per_minute
        self.global_cooldown = Settings.AUDIO_GLOBAL_COOLDOWN
        
        self.last_announcement_time = 0
        self.announcement_history = deque() # Stores (timestamp, priority) for rate limiting
        self.last_announced_object = {} # {object_id: (last_announcement_time, last_announced_text)}
        self.last_scene_summary_time = 0
        self.last_safe_path_time = 0
        self.current_scene_context = None
        
        self.cooldown_timers = {} # {object_id: last_announcement_time}

    def process_feedback(self, obstacles: list[ObstacleInfo], scene_context: SceneContext):
        """
        Process obstacles and generate appropriate announcements
        
        Args:
            obstacles: List of ObstacleInfo objects (from obstacle_classifier)
            scene_context: SceneContext object (from scene_context_analyzer)
        
        Returns:
            List of announcements that were made
        """
        self.current_scene_context = scene_context
        announcements_made = []

        # 1. Handle critical obstacles immediately
        critical_obstacles = [obs for obs in obstacles if obs.risk_level == Priority.CRITICAL]
        for obs in critical_obstacles:
            if self._should_announce_obstacle(obs, bypass_cooldown=True):
                self._make_announcement(obs, interrupt=True)
                announcements_made.append(obs)

        # 2. Check global cooldown for non-critical announcements
        if (time.time() - self.last_announcement_time) < self.global_cooldown:
            return announcements_made # Exit if still in cooldown

        # 3. Process non-critical obstacles
        non_critical_obstacles = [obs for obs in obstacles if obs.risk_level != Priority.CRITICAL]
        filtered_obstacles = self._filter_obstacles_by_mode(non_critical_obstacles)

        # Find the most important non-critical obstacle to announce
        if filtered_obstacles:
            # Sort by priority score to find the most important one
            filtered_obstacles.sort(key=lambda x: x.priority_score, reverse=True)
            
            for obs in filtered_obstacles:
                if self._should_announce_obstacle(obs):
                    self._make_announcement(obs)
                    announcements_made.append(obs)
                    break # Announce only the most important one for now
        
        # 4. Handle scene summaries and safe path announcements
        if not obstacles and (time.time() - self.last_safe_path_time > Settings.AUDIO_COOLDOWN_LOW):
            self._announce_safe_path()
            self.last_safe_path_time = time.time()
        
        if Settings.AUDIO_SCENE_SUMMARY_ENABLED and (time.time() - self.last_scene_summary_time > Settings.AUDIO_SCENE_SUMMARY_INTERVAL):
            self._periodic_update(obstacles)
            self.last_scene_summary_time = time.time()

        return announcements_made

    def _filter_obstacles_by_mode(self, obstacles: list[ObstacleInfo]) -> list[ObstacleInfo]:
        """
        Filters obstacles based on the current system feedback mode.
        """
        min_priority_level = {
            SystemMode.NORMAL: Settings.FEEDBACK_NORMAL_MIN_PRIORITY,
            SystemMode.QUIET: Settings.FEEDBACK_QUIET_MIN_PRIORITY,
            SystemMode.VERBOSE: Settings.FEEDBACK_VERBOSE_MIN_PRIORITY,
            SystemMode.SILENT: Priority.IGNORE.value # No announcements in silent mode
        }.get(Settings.FEEDBACK_MODE, Settings.FEEDBACK_NORMAL_MIN_PRIORITY)

        return [obs for obs in obstacles if obs.risk_level.value >= min_priority_level]

    def _make_announcement(self, obs: ObstacleInfo, interrupt: bool = False):
        """Generate and queue an announcement for an obstacle."""
        announcement_text = self._generate_announcement_text(obs)
        if announcement_text:
            priority_settings = VOICE_SETTINGS.get(obs.risk_level, {})
            should_interrupt = interrupt or priority_settings.get('interrupt', False)
            
            self.announcement_queue.add_announcement(
                announcement_text, obs.risk_level, interrupt=should_interrupt, metadata={'obstacle_id': obs.id}
            )
            
            # Update timers
            self.last_announced_object[obs.id] = (time.time(), announcement_text)
            self.cooldown_timers[obs.id] = time.time()
            self.last_announcement_time = time.time()
            self.announcement_history.append((time.time(), obs.risk_level))

    def _should_announce_obstacle(self, obstacle: ObstacleInfo, bypass_cooldown: bool = False) -> bool:
        """
        Determine if obstacle should be announced based on various rules.
        
        Args:
            obstacle: The obstacle to check.
            bypass_cooldown: If True, ignore global and object-specific cooldowns.
        """
        if Settings.FEEDBACK_MODE == SystemMode.SILENT:
            return False

        # For critical obstacles, we can bypass cooldowns
        if bypass_cooldown and obstacle.risk_level == Priority.CRITICAL:
            # Still check if we just announced the *exact same thing*
            last_time, last_text = self.last_announced_object.get(obstacle.id, (0, ""))
            if time.time() - last_time < 1.0: # Avoid rapid-fire repeats of the same critical alert
                return False
            return True

        # Check global rate limiting
        self._clean_announcement_history()
        if len(self.announcement_history) >= self.max_announcements_per_minute:
            return False

        # Check object-specific cooldown
        last_announcement_time = self.cooldown_timers.get(obstacle.id, 0)
        cooldown_period = ObjectPriorityRules.get_cooldown(obstacle.risk_level)
        if (time.time() - last_announcement_time) < cooldown_period:
            return False

        # --- Logic for announcing significant changes ---

        # 1. Is it a new object?
        if obstacle.id not in self.last_announced_object:
            return True

        # 2. Has its state changed significantly?
        last_time, last_text = self.last_announced_object[obstacle.id]
        
        # Generate new text to see if it's different
        new_text = self._generate_announcement_text(obstacle)
        
        # Announce if the description has changed (e.g., "near" to "immediate")
        if new_text != last_text:
            return True

        return False # Default to not re-announcing

    def _generate_announcement_text(self, obstacle: ObstacleInfo) -> str:
        """
        Generate natural language announcement for an obstacle.
        """
        template_key = 'obstacle_near'
        if obstacle.risk_level == Priority.CRITICAL:
            template_key = 'immediate_danger'
        elif obstacle.spatial_info.is_moving:
            template_key = 'moving_object'
        
        template = ANNOUNCEMENT_TEMPLATES.get(template_key, "Object detected: {object}")
        
        object_name = obstacle.spatial_info.detection.class_name
        distance = obstacle.spatial_info.distance
        direction = obstacle.spatial_info.direction.value if obstacle.spatial_info.direction else ""
        
        return template.format(
            object=object_name,
            distance=f"{distance:.1f}" if distance is not None else "unknown",
            direction=direction
        )

    def _announce_safe_path(self):
        """
        Announce when path is clear.
        """
        if Settings.FEEDBACK_MODE != SystemMode.SILENT:
            self.announcement_queue.add_announcement(
                ANNOUNCEMENT_TEMPLATES['safe_path'], Priority.LOW
            )

    def _periodic_update(self, obstacles: list[ObstacleInfo]):
        """
        Provide periodic scene summary.
        """
        if Settings.FEEDBACK_MODE == SystemMode.VERBOSE and obstacles:
            object_names = [obs.spatial_info.detection.class_name for obs in obstacles]
            summary_text = ANNOUNCEMENT_TEMPLATES['scene_summary'].format(
                count=len(object_names), objects=", ".join(object_names[:3])
            )
            self.announcement_queue.add_announcement(summary_text, Priority.LOW)

    def announce_system_state(self, message: str):
        """
        Announce a system state change (e.g., paused, resumed, mode change).
        """
        if Settings.FEEDBACK_MODE != SystemMode.SILENT:
            self.announcement_queue.add_announcement(
                message, Priority.LOW, interrupt=False # Don't interrupt for simple state changes
            )

    def _clean_announcement_history(self):
        """
        Remove old entries from announcement history to maintain rate limit window.
        """
        one_minute_ago = time.time() - 60
        while self.announcement_history and self.announcement_history[0][0] < one_minute_ago:
            self.announcement_history.popleft()

    def announce_critical_immediately(self, obstacle: ObstacleInfo):
        """
        Bypass queue for critical obstacles and interrupt current speech if necessary.
        """
        announcement_text = self._generate_announcement_text(obstacle)
        if announcement_text:
            self.announcement_queue.add_announcement(
                announcement_text, Priority.CRITICAL, interrupt=True, metadata={'obstacle_id': obstacle.id}
            )
            self.last_announced_object[obstacle.id] = (time.time(), announcement_text)
            self.cooldown_timers[obstacle.id] = time.time()
