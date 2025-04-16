FROM python:3.10

# 🧱 System dependencies (fixes all build-related errors)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    ffmpeg \
    libsndfile1 \
    python3.10-distutils \
    python3.10-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 🛠️ Create working directory
WORKDIR /app

# 🚚 Copy code
COPY . .

# ⬆️ Upgrade tools
RUN pip install --upgrade pip setuptools wheel

# 📦 Install Python dependencies
RUN pip install -r requirements.txt

# 🚀 Run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
