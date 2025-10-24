import pyttsx3
import threading
import os
from enum import Enum

class TTSEngineType(Enum):
    PYTTSX3 = "pyttsx3"
    # Add other TTS engines here if needed

class TTSEngine:
    def __init__(self, engine_type=TTSEngineType.PYTTSX3, voice_rate=150, volume=1.0, voice_id=None):
        self.engine_type = engine_type
        self.voice_rate = voice_rate
        self.volume = volume
        self.voice_id = voice_id
        self._engine = None
        self._lock = threading.Lock()
        self._initialized = False

        self._initialize_engine()

    def _initialize_engine(self):
        if self._initialized:
            return

        print(f"Initializing TTS engine: {self.engine_type.value}...")
        if self.engine_type == TTSEngineType.PYTTSX3:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self.voice_rate)
                self._engine.setProperty('volume', self.volume)

                if self.voice_id:
                    self._engine.setProperty('voice', self.voice_id)
                else:
                    # Attempt to find a female voice if not specified
                    voices = self._engine.getProperty('voices')
                    female_voices = [voice.id for voice in voices if 'female' in voice.name.lower()]
                    if female_voices:
                        self._engine.setProperty('voice', female_voices[0])
                        print(f"Selected female voice: {self._engine.getProperty('voice')}")
                    else:
                        print("No female voice found, using default.")
                
                self._initialized = True
                print("TTS engine initialized.")
            except Exception as e:
                print(f"Error initializing pyttsx3: {e}")
                self._engine = None
        else:
            print(f"Unsupported TTS engine type: {self.engine_type}")
        
        if not self._initialized:
            print("TTS engine failed to initialize. Audio feedback will be disabled.")

    def say(self, text):
        with self._lock:
            if not self._initialized:
                # Attempt to re-initialize if it failed previously
                self._initialize_engine()
            
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"Error speaking text: {e}")
            else:
                print(f"TTS engine not available, cannot say: {text}")

    def set_voice_rate(self, rate):
        self.voice_rate = rate
        if self._engine:
            self._engine.setProperty('rate', self.voice_rate)

    def set_volume(self, volume):
        self.volume = volume
        if self._engine:
            self._engine.setProperty('volume', self.volume)

    def set_voice(self, voice_id):
        self.voice_id = voice_id
        if self._engine:
            self._engine.setProperty('voice', self.voice_id)
            print(f"Voice set to: {self._engine.getProperty('voice')}")

    def get_available_voices(self):
        if not self._initialized:
            self._initialize_engine()
        if self._engine:
            voices = self._engine.getProperty('voices')
            return [{'id': voice.id, 'name': voice.name, 'gender': voice.gender, 'age': voice.age} for voice in voices]
        return []

    def stop(self):
        if self._engine:
            self._engine.stop()
            print("TTS engine stopped.")

    def stop_speaking(self):
        """Immediately stops any currently speaking audio."""
        if self._engine:
            self._engine.stop()
