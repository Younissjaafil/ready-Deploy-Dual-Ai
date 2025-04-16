FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    ffmpeg \
    libsndfile1 \
    python3-distutils \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Upgrade pip + prevent wheels from building from source
RUN pip install --upgrade pip setuptools wheel
RUN pip install --prefer-binary --no-cache-dir numpy==1.22.0 scipy scikit-learn

# Then install TTS and everything else
RUN pip install -r requirements.txt

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
