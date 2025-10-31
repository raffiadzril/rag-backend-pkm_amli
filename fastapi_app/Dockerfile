FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set pip timeout and other configurations
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_TIMEOUT=100
ENV PIP_RETRIES=3

# Copy requirements first for better caching
COPY requirements.txt .

# Install FastAPI dependencies first
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Install core dependencies separately to handle timeouts
RUN pip install --no-cache-dir --default-timeout=100 google-generativeai python-dotenv requests

# Install LangChain dependencies separately
RUN pip install --no-cache-dir --default-timeout=100 langchain-community

# Install sentence transformers and chroma separately
RUN pip install --no-cache-dir --default-timeout=100 sentence-transformers

# Install chromadb separately
RUN pip install --no-cache-dir --default-timeout=100 chromadb

# Install langchain-huggingface and langchain-chroma
RUN pip install --no-cache-dir --default-timeout=100 langchain-huggingface langchain-chroma

# Copy the entire project structure to maintain all dependencies
COPY . .

# Set working directory to fastapi_app
WORKDIR /app/fastapi_app

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]