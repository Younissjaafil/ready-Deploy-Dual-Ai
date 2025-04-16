FROM continuumio/miniconda3

# Create environment
RUN conda create -n dualai python=3.10 -y

# Activate conda + install deps
SHELL ["conda", "run", "-n", "dualai", "/bin/bash", "-c"]

RUN conda install -y \
    numpy=1.22 \
    scikit-learn \
    pip \
    ffmpeg \
    && pip install \
    fastapi==0.104.1 \
    uvicorn==0.23.2 \
    pydub==0.25.1 \
    torch==2.2.0 \
    diffusers==0.24.0 \
    transformers==4.35.2 \
    TTS==0.17.2 \
    Pillow==10.0.1

# Copy app
WORKDIR /app
COPY . .

# Expose and run
CMD ["conda", "run", "--no-capture-output", "-n", "dualai", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
