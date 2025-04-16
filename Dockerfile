FROM python:3.10

# 📦 Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    python3-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📁 Set working directory
WORKDIR /app

# 🚚 Copy project files
COPY . .

# ⬆️ Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# ✅ Preinstall numpy + sklearn as binary (avoid build)
RUN pip install numpy==1.22.0 scikit-learn==1.1.3 --only-binary=:all:

# ✅ Now install everything else including TTS
RUN pip install -r requirements.txt

# 🚀 Launch FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
