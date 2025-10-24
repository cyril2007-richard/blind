from queue import PriorityQueue
import threading
import time
from enum import Enum

from config.priority_rules import Priority, VOICE_SETTINGS

class AnnouncementQueue:
    """
    Thread-safe priority queue for announcements
    Ensures critical announcements are spoken first
    """
    
    def __init__(self, max_size=50):
        """
        Args:
            max_size: Maximum queue size (older items dropped if full)
        """
        self.queue = PriorityQueue(maxsize=max_size)
        self.is_running = False
        self.worker_thread = None
        self.tts_engine = None
        self.current_announcement = None
        self.lock = threading.Lock()
    
    def add_announcement(self, text, priority, interrupt=False, metadata=None):
        """
        Add announcement to queue
        
        Args:
            text: Announcement text
            priority: Priority enum (CRITICAL, HIGH, MEDIUM, LOW)
            interrupt: If True, clear queue and speak immediately
            metadata: Additional info (obstacle type, distance, etc.)
        
        Returns:
            bool: True if added successfully
        """
        with self.lock:
            if interrupt:
                self.clear_queue()
                # If an announcement is currently speaking, try to stop it
                if self.tts_engine and hasattr(self.tts_engine, 'stop_speaking'):
                    self.tts_engine.stop_speaking()

            # Check for duplicate announcements
            for _, _, (existing_text, _) in self.queue.queue:
                if existing_text == text:
                    return False # Announcement already exists
            
            # PriorityQueue is min-heap, so we negate priority for max-heap behavior
            # Tuple (priority_value, timestamp, announcement_data)
            # Timestamp ensures FIFO for same priority
            announcement_data = (text, metadata)
            self.queue.put((-priority.value, time.time(), announcement_data))
            return True
    
    def start_processing(self, tts_engine):
        """Start background thread to process queue"""
        if self.is_running:
            return
        self.tts_engine = tts_engine
        self.tts_engine = tts_engine
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
    
    def stop_processing(self):
        """Stop queue processing"""
        if not self.is_running:
            return
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0) # Give it a moment to finish
            if self.worker_thread.is_alive():
                print("Warning: AnnouncementQueue worker thread did not terminate gracefully.")
        print("AnnouncementQueue processing stopped.")
    
    def clear_queue(self):
        """Clear all pending announcements (for critical interrupts)"""
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass # Queue was already empty
            print("AnnouncementQueue cleared.")
    
    def get_queue_size(self):
        """Get number of pending announcements"""
        return self.queue.qsize()
    
    def _process_loop(self):
        """
        Background worker that processes queue
        Runs in separate thread
        """
        while self.is_running:
            try:
                # Get item with timeout to allow thread to check is_running
                neg_priority, timestamp, (text, metadata) = self.queue.get(timeout=0.1)
                priority = Priority(-neg_priority) # Convert back to positive priority
                
                self.current_announcement = (text, priority, metadata)
                
                if not self.tts_engine:
                    print("TTS engine not available in AnnouncementQueue._process_loop.")
                    self.queue.task_done()
                    continue

                # Apply voice settings based on priority
                voice_settings = VOICE_SETTINGS.get(priority, {})
                original_rate = self.tts_engine.voice_rate
                original_volume = self.tts_engine.volume

                if 'rate' in voice_settings:
                    self.tts_engine.set_voice_rate(original_rate * voice_settings['rate'])
                if 'volume' in voice_settings:
                    self.tts_engine.set_volume(original_volume * voice_settings['volume'])

                self.tts_engine.say(text)

                # Restore original voice settings
                self.tts_engine.set_voice_rate(original_rate)
                self.tts_engine.set_volume(original_volume)

                self.current_announcement = None
                self.queue.task_done()
            except Exception as e:
                if self.is_running: # Only log if not intentionally stopping
                    print(f"Error in AnnouncementQueue processing loop: {e}")
            time.sleep(0.01) # Small sleep to prevent busy-waiting
