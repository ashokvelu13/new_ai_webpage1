import io
import os
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

BASE_DIR = os.path.dirname(__file__)

# Smart Path Resolver: Checks for different naming variants to guarantee Linux loading
def resolve_file_path(filename_options: list) -> str:
    for filename in filename_options:
        full_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(full_path):
            return full_path
    return os.path.join(BASE_DIR, filename_options[0])

# Match whatever names exist on your repository disk
ONNX_MODEL_PATH = resolve_file_path(["kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx", "Kokoro-v1.0.onnx"])
VOICES_BIN_PATH = resolve_file_path(["voices-v1.0.bin", "Voices-v1.0.bin", "voices.bin"])

# Initialize the ONNX Runtime context safely
if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(VOICES_BIN_PATH):
    print(f"CRITICAL MODEL PATH FAIL - Checking: {ONNX_MODEL_PATH} and {VOICES_BIN_PATH}")
    kokoro = None
else:
    print(f"Successfully located target files. Loading model: {ONNX_MODEL_PATH}")
    try:
        kokoro = Kokoro(ONNX_MODEL_PATH, VOICES_BIN_PATH)
        print("Kokoro ONNX TTS Engine operational!")
    except Exception as e:
        print(f"Failed to bootstrap engine weights: {str(e)}")
        kokoro = None


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Reads the static index.html file and serves it to the browser."""
    html_path = os.path.join(BASE_DIR, "index.html")
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
            detail=f"Model files failed to initialize. Active ONNX Path: {ONNX_MODEL_PATH}"
        )
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    # Dynamically track matching lang patterns
    lang_code = "en-us"
    if voice.startswith("af_") or voice.startswith("am_"):
        lang_code = "en-us"
    elif voice.startswith("bf_") or voice.startswith("bm_"):
        lang_code = "en-gb"
    elif voice.startswith("jf_"):
        lang_code = "ja"

    try:
        # Check if the requested voice string exists in the model keys map
        available_voices = kokoro.get_voices()
        if voice not in available_voices:
            raise HTTPException(
                status_code=400, 
                detail=f"Voice '{voice}' missing. Loaded options are: {', '.join(available_voices[:5])}..."
            )
    except Exception:
        pass

    try:
        samples, sample_rate = kokoro.create(
            text=text, 
            voice=voice, 
            speed=1.0, 
            lang=lang_code
        )

        # Stream binary audio out safely
        wav_io = io.BytesIO()
        sf.write(wav_io, samples, sample_rate, format='WAV')
        wav_io.seek(0)

        return StreamingResponse(wav_io, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Run Failure: {str(e)}")
