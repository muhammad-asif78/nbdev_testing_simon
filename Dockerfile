FROM python:3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install -r requirements.txt 
    
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         git \
#         build-essential \
#         libgl1 \
#         libglib2.0-0 \
#         libsm6 \
#         libxext6 \
#         libxrender-dev \
#         poppler-utils \
#         wget \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt && \
#     pip install paddlepaddle==2.6.1

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]