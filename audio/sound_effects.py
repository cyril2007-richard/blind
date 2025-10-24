class SoundEffects:
    """
    Generates and plays non-verbal audio cues
    Useful for quick alerts without speech
    """
    
    def __init__(self, sample_rate=44100):
        """Initialize audio system"""
        pass
    
    def play_warning_beep(self, urgency='medium'):
        """
        Play warning beep
        
        Args:
            urgency: 'low', 'medium', 'high', 'critical'
                    (affects frequency and pattern)
        """
        pass
    
    def play_proximity_tone(self, distance):
        """
        Play tone that varies with distance
        Closer = higher frequency/faster beeps
        
        Args:
            distance: Distance in meters
        """
        pass
    
    def play_direction_cue(self, direction):
        """
        Play spatial audio cue indicating direction
        Left ear = left obstacle, right ear = right obstacle
        
        Args:
            direction: 'left', 'right', 'center'
        """
        pass
    
    def play_safe_tone(self):
        """Pleasant tone indicating clear path"""
        pass
    
    def generate_tone(self, frequency, duration, volume=0.5):
        """
        Generate pure tone
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            volume: Volume 0.0 to 1.0
        
        Returns:
            numpy array of audio samples
        """
        pass