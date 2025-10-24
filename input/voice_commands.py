class VoiceCommandListener:
    """
    Listens for voice commands from user
    Supports basic commands like "stop", "repeat", "where am I"
    """
    
    def __init__(self, wake_word="hey assistant", language="en-US"):
        """
        Args:
            wake_word: Activation phrase before commands
            language: Speech recognition language
        """
        self.wake_word = wake_word.lower()
        self.language = language
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        self.command_callback = None
        pass
    
    def start_listening(self, callback):
        """
        Start background listening for commands
        
        Args:
            callback: Function to call when command recognized
                     callback(command: str)
        """
        pass
    
    def stop_listening(self):
        """Stop listening for commands"""
        pass
    
    def _listen_loop(self):
        """
        Background thread that continuously listens
        Runs speech recognition on audio segments
        """
        pass
    
    def recognize_command(self, audio_data):
        """
        Convert audio to text command
        
        Args:
            audio_data: Audio segment
        
        Returns:
            str: Recognized command or None
        """
        pass
    
    def process_command(self, command_text):
        """
        Parse and execute command
        
        Commands:
            - "stop" / "pause": Pause announcements
            - "resume" / "start": Resume announcements
            - "repeat": Repeat last announcement
            - "where am I": Provide scene summary
            - "what's ahead": List obstacles in front
            - "quiet mode": Reduce announcements
            - "verbose mode": Increase announcements
        
        Args:
            command_text: Recognized text
        
        Returns:
            dict: {action: str, params: dict}
        """
        pass