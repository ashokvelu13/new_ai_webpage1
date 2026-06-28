import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# 1. Configuration Settings
FS = 16000          # Sample rate required by Whisper (16kHz)
DURATION = 5        # How long to record (in seconds)
OUTPUT_FILENAME = "temp_recording.wav"

# Choose model size: 'tiny', 'base', 'small', 'medium', or 'large-v3'
# 'base' or 'small' are perfect for local PCs (fast and accurate)
MODEL_SIZE = "base" 

print("⏳ Loading local Whisper Speech-to-Text Model...")
# Set device="cpu" or device="cuda" if you have a dedicated NVIDIA GPU
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def record_audio():
    """Records audio from your microphone."""
    print(f"\n🎙️ Recording started for {DURATION} seconds... Speak now!")
    
    # Record audio data as a float32 array
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording time finishes
    
    print("🛑 Recording finished. Processing audio...")
    
    # Save the recorded audio to a temporary WAV file
    write(OUTPUT_FILENAME, FS, (recording * 32767).astype(np.int16))

def transcribe_audio():
    """Transcribes the saved audio file into text."""
    if not os.path.exists(OUTPUT_FILENAME):
        print("❌ Error: Recording file not found.")
        return

    # Whisper automatically detects if you are speaking Tamil or English!
    segments, info = model.transcribe(OUTPUT_FILENAME, beam_size=5)
    
    print(f"🌍 Detected Language: {info.language} (Confidence: {info.language_probability:.2f})")
    print("\n📝 Transcribed Text:")
    
    full_text = ""
    for segment in segments:
        print(segment.text)
        full_text += segment.text
        
    # Clean up the temporary audio file
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
        
    return full_text

if __name__ == "__main__":
    print("--- Local Speech-to-Text Engine ---")
    while True:
        user_choice = input("\nPress Enter to start recording (or type 'exit' to quit): ")
        if user_choice.lower() in ['exit', 'quit']:
            break
            
        record_audio()
        transcribe_audio()
