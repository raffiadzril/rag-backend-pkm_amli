#-------------------------------------------------------------------
# STAGE 1: The "Builder"
#
# This stage installs system tools and ALL Python packages into a
# virtual environment. This entire stage will be cached.
#-------------------------------------------------------------------
FROM python:3.10-slim as builder

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set pip configurations
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_TIMEOUT=100
ENV PIP_RETRIES=3

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy ONLY the requirements file (from fastapi_app subfolder)
COPY fastapi_app/requirements.txt .

# Install all dependencies into the venv
# This is the slow step that will be cached
RUN pip install --no-cache-dir -r requirements.txt


#-------------------------------------------------------------------
# STAGE 2: The "Final" Image
#
# This stage is clean. It just copies the virtual env from the
# builder and your application code. It's fast and small.
#-------------------------------------------------------------------
FROM python:3.10-slim as final

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy your application code
COPY fastapi_app/ ./fastapi_app/
COPY dataset/ /dataset/
COPY rag-system/query.py ./

# Set paths to use the venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Set the final working directory
WORKDIR /app/fastapi_app

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]