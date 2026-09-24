FROM python:3.11-slim

# ffmpeg + libgl/glib needed by OpenCV video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Core deps only by default — this keeps the image small. Install torch
# separately (see README) matching your target platform/CUDA version, or
# uncomment the line below for CPU-only torch in the image itself.
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic PyYAML python-multipart \
        numpy opencv-python-headless pytest httpx

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
