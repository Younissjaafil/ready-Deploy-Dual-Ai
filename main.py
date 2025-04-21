# # main.py
# import os, uuid, random
# from typing import Optional
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse, JSONResponse
# from pydub import AudioSegment
# import torch
# import numpy as np
# from PIL import Image
# from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
# from TTS.api import TTS
# from torch.serialization import safe_globals
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
# from TTS.config.shared_configs import BaseDatasetConfig

# # --- Setup ---
# os.environ["COQUI_TOS_AGREED"] = "1"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# MAX_SEED = np.iinfo(np.int32).max

# app = FastAPI()

# # --- Health Check ---
# @app.get("/")
# def read_root():
#     return {"status": "running"}

# # --- Load Models ---
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "fluently/Fluently-XL-Final",
#     torch_dtype=torch.float16,
#     use_safetensors=True
# )
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.load_lora_weights("ehristoforu/dalle-3-xl-v2", weight_name="dalle-3-xl-lora-v2.safetensors", adapter_name="dalle")
# pipe.set_adapters("dalle")
# pipe.to(device)

# # with safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
# #     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# with safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
#     tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

# os.makedirs("voices", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)

# # --- Image Generation Endpoint ---
# @app.post("/generate-image")
# async def generate_image(
#     prompt: str = Form(...),
#     negative_prompt: Optional[str] = Form(""),
#     use_negative_prompt: Optional[bool] = Form(True),
#     seed: Optional[int] = Form(0),
#     randomize_seed: Optional[bool] = Form(True),
#     width: Optional[int] = Form(1024),
#     height: Optional[int] = Form(1024),
#     guidance_scale: Optional[float] = Form(6.0),
# ):
#     seed = random.randint(0, MAX_SEED) if randomize_seed else seed
#     if not use_negative_prompt:
#         negative_prompt = ""

#     result = pipe(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         width=width,
#         height=height,
#         guidance_scale=guidance_scale,
#         num_inference_steps=25,
#         num_images_per_prompt=1,
#         cross_attention_kwargs={"scale": 0.65},
#         output_type="pil"
#     ).images

#     output_path = f"{uuid.uuid4().hex}.jpg"
#     result[0].save(output_path)
#     return FileResponse(output_path)

# # --- Voice Cloning Endpoint ---
# @app.post("/clone-voice")
# async def clone_voice(
#     text: str = Form(...),
#     language: str = Form(...),
#     audio: UploadFile = File(...)
# ):
#     if not audio:
#         return JSONResponse({"error": "No audio file uploaded"}, status_code=400)

#     os.makedirs("uploads", exist_ok=True)
#     input_path = f"./uploads/{uuid.uuid4().hex}_{audio.filename}"
#     with open(input_path, "wb") as f:
#         f.write(await audio.read())

#     if input_path.endswith(".mp3"):
#         converted = input_path.replace(".mp3", ".wav")
#         AudioSegment.from_mp3(input_path).export(converted, format="wav")
#         input_path = converted

#     output_path = "./output.wav"
#     tts.tts_to_file(text=text, speaker_wav=input_path, language=language, file_path=output_path)
#     return FileResponse(output_path)


# @app.post("/record_voice")
# async def record_voice(audio: UploadFile, user_id: str = Form(...)):
#     voice_path = f"./voices/{user_id}.wav"
#     audio_bytes = await audio.read()
#     temp_path = f"./voices/temp_input.wav"
#     with open(temp_path, "wb") as f:
#         f.write(audio_bytes)
#     AudioSegment.from_file(temp_path).export(voice_path, format="wav")
#     os.remove(temp_path)
#     return {"message": f"Voice saved for ID: {user_id}"}

 

# @app.post("/speak_caption")
# async def generate_voice(user_id: str = Form(...), text: str = Form(...)):
#     voice_path = f"./voices/{user_id}.wav"
#     if not os.path.exists(voice_path):
#         return {"error": "Voice sample not found."}

#     output_path = f"./outputs/{user_id}_{abs(hash(text)) % 100000}_out.wav"
#     tts.tts_to_file(
#         text=text,
#         speaker_wav=voice_path,
#         language="en",
#         file_path=output_path
#     )

#     return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path), headers={
#         "Content-Disposition": "inline"
#     })

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
import uvicorn

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
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

os.makedirs("voices", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --- Image Generation Endpoint ---
@app.post("/generate-image")
# async def generate_image(
#     prompt: str = Form(...),
#     negative_prompt: Optional[str] = Form(""),
#     use_negative_prompt: Optional[bool] = Form(True),
#     seed: Optional[int] = Form(0),
#     randomize_seed: Optional[bool] = Form(True),
#     width: Optional[int] = Form(1024),
#     height: Optional[int] = Form(1024),
#     guidance_scale: Optional[float] = Form(6.0),
# ):
    # seed = random.randint(0, MAX_SEED) if randomize_seed else seed
    # if not use_negative_prompt:
    #     negative_prompt = ""

    # result = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     width=width,
    #     height=height,
    #     guidance_scale=guidance_scale,
    #     num_inference_steps=25,
    #     num_images_per_prompt=1,
    #     cross_attention_kwargs={"scale": 0.65},
    #     output_type="pil"
    # ).images

    # output_path = f"{uuid.uuid4().hex}.jpg"
    # result[0].save(output_path)
    # return FileResponse(output_path)

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

# --- Record Voice Endpoint ---
@app.post("/record_voice")
async def record_voice(audio: UploadFile, user_id: str = Form(...)):
    voice_path = f"./voices/{user_id}.wav"
    audio_bytes = await audio.read()
    temp_path = f"./voices/temp_input.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    AudioSegment.from_file(temp_path).export(voice_path, format="wav")
    os.remove(temp_path)
    return {"message": f"Voice saved for ID: {user_id}"}

# --- Speak Caption Endpoint ---
@app.post("/speak_caption")
async def generate_voice(user_id: str = Form(...), text: str = Form(...)):
    voice_path = f"./voices/{user_id}.wav"
    if not os.path.exists(voice_path):
        return JSONResponse({"error": "Voice sample not found."}, status_code=404)

    output_path = f"./outputs/{user_id}_{abs(hash(text)) % 100000}_out.wav"
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_path,
        language="en",
        file_path=output_path
    )

    return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path), headers={
        "Content-Disposition": "inline"
    })

# --- Run as HTTP service ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
