# # main.py

# import os
# import uuid
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydub import AudioSegment
# import torch
# from TTS.api import TTS

# # --- Setup ---
# os.environ["COQUI_TOS_AGREED"] = "1"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create necessary directories
# os.makedirs("voices", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)
# os.makedirs("uploads", exist_ok=True)

# # TTS model is loaded only when needed
# tts = None

# def get_tts_model():
#     global tts
#     if tts is None:
#         print("Loading TTS model...")
#         tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
#                   gpu=torch.cuda.is_available())
#         print("TTS model loaded successfully")
#     return tts

# # --- Health Check ---
# @app.get("/")
# def read_root():
#     return {"status": "healthy", "device": device}

# # --- Voice Cloning Endpoint ---
# @app.post("/clone-voice")
# async def clone_voice(
#     text: str = Form(...),
#     language: str = Form("en"),
#     audio: UploadFile = File(...)
# ):
#     if not audio:
#         return JSONResponse({"error": "No audio file uploaded"}, status_code=400)

#     input_path = f"./uploads/{uuid.uuid4().hex}_{audio.filename}"
#     with open(input_path, "wb") as f:
#         f.write(await audio.read())

#     # Convert mp3 to wav if needed
#     if input_path.endswith(".mp3"):
#         converted = input_path.replace(".mp3", ".wav")
#         AudioSegment.from_mp3(input_path).export(converted, format="wav")
#         input_path = converted

#     output_path = f"./outputs/{uuid.uuid4().hex}_output.wav"
#     model = get_tts_model()
#     model.tts_to_file(text=text, speaker_wav=input_path, language=language, file_path=output_path)
    
#     return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

# # --- Alternative naming for the endpoint ---
# @app.post("/clone_voice")
# async def clone_voice_alt(text: str = Form(...), file: UploadFile = File(...), language: str = Form("en")):
#     input_wav_path = f"./uploads/{uuid.uuid4().hex}_{file.filename}"
#     with open(input_wav_path, "wb") as f:
#         f.write(await file.read())

#     output_path = f"./outputs/{uuid.uuid4().hex}_output.wav"
#     model = get_tts_model()
#     model.tts_to_file(
#         text=text,
#         speaker_wav=input_wav_path,
#         language=language,
#         file_path=output_path
#     )

#     return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

# # --- Record Voice Endpoint ---
# @app.post("/record_voice")
# async def record_voice(audio: UploadFile = File(...), user_id: str = Form(...)):
#     voice_path = f"./voices/{user_id}.wav"
#     audio_bytes = await audio.read()
#     temp_path = f"./voices/temp_input.wav"
#     with open(temp_path, "wb") as f:
#         f.write(audio_bytes)
#     AudioSegment.from_file(temp_path).export(voice_path, format="wav")
#     os.remove(temp_path)
#     return {"message": f"Voice saved for ID: {user_id}"}

# # --- Speak Caption Endpoint ---
# @app.post("/speak_caption")
# async def generate_voice(user_id: str = Form(...), text: str = Form(...), language: str = Form("en")):
#     voice_path = f"./voices/{user_id}.wav"
#     if not os.path.exists(voice_path):
#         return JSONResponse({"error": "Voice sample not found."}, status_code=404)

#     output_path = f"./outputs/{user_id}_{abs(hash(text)) % 100000}_out.wav"
#     model = get_tts_model()
#     model.tts_to_file(
#         text=text,
#         speaker_wav=voice_path,
#         language=language,
#         file_path=output_path
#     )

#     return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

# # --- Run as HTTP service ---
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
# main.py

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can later restrict this to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts = None

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/clone_voice")
async def clone_voice(text: str = Form(...), file: UploadFile = File(...)):
    global tts

    if tts is None:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    input_wav_path = f"/tmp/{file.filename}"
    with open(input_wav_path, "wb") as f:
        f.write(await file.read())

    output_path = "/tmp/output.wav"
    tts.tts_to_file(
        text=text,
        speaker_wav=input_wav_path,
        file_path=output_path
    )

    return {"output": output_path}
