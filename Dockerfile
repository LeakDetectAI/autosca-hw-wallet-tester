FROM python:3.11-slim

WORKDIR /workspace

# Install system deps required by TensorFlow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY requirements.txt setup.py ./

# Pin versions known to work together (TF 2.16 + Keras 3 + AutoKeras 2)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow==2.16.1 \
        keras==3.3.3 \
        keras-tuner==1.4.7 \
        autokeras==2.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Install the package itself
COPY . .
RUN pip install --no-cache-dir -e .

# Default: run GPU check
CMD ["python", "-c", "import deepscapy; print('deepscapy ready'); import tensorflow as tf; print(f'TF {tf.__version__}, GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"]
