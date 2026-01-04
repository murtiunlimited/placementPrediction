# ================================
# Base image (Bookworm has new SQLite)
# ================================
FROM python:3.11-bookworm

# ================================
# Working directory
# ================================
WORKDIR /app

# ================================
# Copy files
# ================================
COPY requirements.txt .
COPY app ./app
COPY cleaning.py .
COPY modeltrain.py .
COPY dataset ./app/dataset
COPY README.md .

# ================================
# Install system deps
# ================================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Install Python deps
# ================================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Streamlit config
# ================================
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# ================================
# Run app
# ================================
CMD ["streamlit", "run", "app/app.py"]
