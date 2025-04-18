# main.py
import os, uuid, random
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from TTS.api import TTS
from torch.serialization import safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# --- Setup ---
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max

app = FastAPI()

# --- Health Check ---
@app.get("/")
def read_root():
    return {"status": "running"}

# --- Load Models ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    "fluently/Fluently-XL-Final",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("ehristoforu/dalle-3-xl-v2", weight_name="dalle-3-xl-lora-v2.safetensors", adapter_name="dalle")
pipe.set_adapters("dalle")
pipe.to(device)

with safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# --- Image Generation Endpoint ---
@app.post("/generate-image")
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(""),
    use_negative_prompt: Optional[bool] = Form(True),
    seed: Optional[int] = Form(0),
    randomize_seed: Optional[bool] = Form(True),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    guidance_scale: Optional[float] = Form(6.0),
):
    seed = random.randint(0, MAX_SEED) if randomize_seed else seed
    if not use_negative_prompt:
        negative_prompt = ""

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil"
    ).images

    output_path = f"{uuid.uuid4().hex}.jpg"
    result[0].save(output_path)
    return FileResponse(output_path)

# --- Voice Cloning Endpoint ---
@app.post("/clone-voice")
async def clone_voice(
    text: str = Form(...),
    language: str = Form(...),
    audio: UploadFile = File(...)
):
    if not audio:
        return JSONResponse({"error": "No audio file uploaded"}, status_code=400)

    os.makedirs("uploads", exist_ok=True)
    input_path = f"./uploads/{uuid.uuid4().hex}_{audio.filename}"
    with open(input_path, "wb") as f:
        f.write(await audio.read())

    if input_path.endswith(".mp3"):
        converted = input_path.replace(".mp3", ".wav")
        AudioSegment.from_mp3(input_path).export(converted, format="wav")
        input_path = converted

    output_path = "./output.wav"
    tts.tts_to_file(text=text, speaker_wav=input_path, language=language, file_path=output_path)
    return FileResponse(output_path)
