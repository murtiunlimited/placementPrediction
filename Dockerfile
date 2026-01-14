# Use official Python slim image for smaller size
FROM python:3.11-slim

# Expose Streamlit port
EXPOSE 8501

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY . /app
RUN pip3 install -r requirements.txt

# Default command to run the app
CMD ["streamlit", "run", "app/app.py"]

