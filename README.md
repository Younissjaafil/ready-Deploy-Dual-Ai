# Dual AI Backend (Image + Voice Generator)

This project provides a FastAPI-based backend that:

- 🎨 Generates images from text prompts using Stable Diffusion XL + LoRA
- 🎧 Clones voices using XTTS v2 from Coqui TTS

## Endpoints

### `/generate-image`

POST → Generates a high-quality image from text

### `/clone-voice`

POST → Clones voice from uploaded `.mp3` or `.wav` and text

## Deployment

### Run Locally

```bash
uvicorn main:app --reload
```
