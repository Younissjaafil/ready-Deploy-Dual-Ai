# FROM python:3.10-slim

# WORKDIR /app

# # System dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsndfile1 \
#     && rm -rf /var/lib/apt/lists/*

# # Python dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # App code
# COPY . .

# # Make directories that may be needed
# RUN mkdir -p voices uploads outputs

# # Set port from environment or default to 8000
# ENV PORT=8000
# EXPOSE 8000

# # Command to run the app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# App code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
