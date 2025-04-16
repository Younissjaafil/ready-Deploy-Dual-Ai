FROM ghcr.io/coqui-ai/tts-cpu:latest

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir fastapi uvicorn pydub Pillow

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
