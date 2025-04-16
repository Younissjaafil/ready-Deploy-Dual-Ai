FROM python:3.10

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    python3-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Expose and run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

