import io
import os
import urllib.request
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro

app = FastAPI(title="TTS Player")

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target model file paths - Using int8 version to prevent Render from running out of RAM memory
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "kokoro-v1.0.int8.onnx")
VOICES_BIN_PATH = os.path.join(os.path.dirname(__file__), "voices-v1.0.bin")

def ensure_model_files_exist():
    """Checks if the required AI weight files exist. Downloads them if missing."""
    base_url = "https://github.com"
    
    # 1. Check and download ONNX model file
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Downloading {ONNX_MODEL_PATH} from GitHub releases... Please wait.")
        try:
            urllib.request.urlretrieve(f"{base_url}kokoro-v1.0.int8.onnx", ONNX_MODEL_PATH)
            print("ONNX model download complete!")
        except Exception as e:
            print(f"Failed to download ONNX file: {e}")

    # 2. Check and download Voices configuration file
    if not os.path.exists(VOICES_BIN_PATH):
        print(f"Downloading {VOICES_BIN_PATH} from GitHub releases... Please wait.")
        try:
            urllib.request.urlretrieve(f"{base_url}voices-v1.0.bin", VOICES_BIN_PATH)
            print("Voices configuration binary download complete!")
        except Exception as e:
            print(f"Failed to download Voices file: {e}")

# Run the downloader verification checklist before initializing the runtime engine
ensure_model_files_exist()

# Initialize the ONNX Runtime context
if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(VOICES_BIN_PATH):
    print("CRITICAL WARNING: Model files could not be acquired. TTS features will be offline!")
    kokoro = None
else:
    # Instantiates directly from your local assets/downloaded data paths
    print("Initializing Kokoro ONNX Text-to-Speech Engine...")
    kokoro = Kokoro(ONNX_MODEL_PATH, VOICES_BIN_PATH)
    print("TTS engine successfully loaded and ready!")


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Reads the static index.html file and serves it to the browser."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html file not found in directory.")
        
    with open(html_path, "r", encoding="utf-8") as file:
        return file.read()


@app.post("/tts")
async def text_to_speech(text: str = Form(...), voice: str = Form("af_heart")):
    """Accepts text, maps the local ONNX model, and streams the output stream."""
    if not kokoro:
        raise HTTPException(
            status_code=500, 
            detail="ONNX model or voices binary file is missing from server workspace directory."
        )
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    try:
        # Generate the raw audio samples array using ONNX runtime
        # lang="en-us" matches American/British voice profiles by default
        samples, sample_rate = kokoro.create(
            text=text, 
            voice=voice, 
            speed=1.0, 
            lang="en-us"
        )

        # Write WAV binary stream directly into memory
        wav_io = io.BytesIO()
        sf.write(wav_io, samples, sample_rate, format='WAV')
        wav_io.seek(0)

        return StreamingResponse(wav_io, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
