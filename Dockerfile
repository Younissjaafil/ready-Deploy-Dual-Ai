FROM python:3.10

# ✅ Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    python3-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📁 Set work directory
WORKDIR /app

# 📝 Copy app code
COPY . .

# 🔧 Upgrade pip + install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# 🚀 Start your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
