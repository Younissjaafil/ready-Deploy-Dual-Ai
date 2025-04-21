# FROM ghcr.io/coqui-ai/tts-cpu:latest

# WORKDIR /app
# COPY . .

# RUN pip install --no-cache-dir fastapi uvicorn pydub Pillow

# EXPOSE 7860
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
FROM ghcr.io/coqui-ai/tts-cpu:latest

WORKDIR /app
COPY . .

# 1) Add python-multipart so Form()/File() works
# 2) Install uvicorn[standard] for a better production server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydub

EXPOSE 7860

# Run your main.py so it binds to 0.0.0.0:$PORT automatically
CMD ["python", "main.py"]
