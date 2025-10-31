# MPASI Menu Generator - Docker Deployment

This project includes a FastAPI application for generating MPASI (Pendamping ASI) menus for babies, with Docker support.

## Prerequisites

- Docker Desktop installed
- Docker Compose installed
- Google API Key for Gemini (optional, if using Gemini model)

## Setup Instructions

### 1. Environment Configuration

Before building the Docker images, you need to set up your environment variables.

Create a `.env` file in the root directory with the following content:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Dataset Preparation

The application requires the following dataset files in the `../dataset` directory relative to the fastapi_app:
- `TKPI-2020.json` - The food composition database
- `akg_merged.json` - Age-specific nutritional requirements
- `aturan-mpasi.json` - MPASI guidelines
- Other markdown files for context

Make sure these files are available before running the application.

### 3. ChromaDB Indexing

Before running the application, you need to have the ChromaDB index. If it doesn't exist, run the store.py script first in the rag-system directory to create the database.

## Running with Docker Compose

To start the entire system:

```bash
docker-compose up -d
```

This will start:
- FastAPI application on port 8000
- ChromaDB vector database on port 8001

## API Endpoints

Once running, the API will be available at `http://localhost:8000`

- `GET /` - Health check
- `GET /api/status` - Service status
- `GET /api/models` - Available models
- `POST /api/generate-menu` - Generate MPASI menu (main endpoint)

### Example Request

```json
{
  "umur_bulan": 6,
  "berat_badan": 7.0,
  "tinggi_badan": 65,
  "jenis_kelamin": "laki-laki",
  "tempat_tinggal": "Indonesia",
  "alergi": ["telur"],
  "model_type": "gemini",
  "model_name": null
}
```

## Building Images

To build the Docker images:

```bash
docker-compose build
```

## Development Mode

For development with hot reloading:

```bash
docker-compose -f docker-compose.dev.yml up
```

Note: A development compose file is not included by default, you may want to create one with volume mounts for development.

## Troubleshooting

1. **ChromaDB Connection Issues**: Ensure the ChromaDB container is running and accessible.
2. **Missing Dataset Files**: Verify that all required dataset files exist in the dataset directory.
3. **Google API Key Issues**: Ensure the GOOGLE_API_KEY environment variable is properly set.
4. **Memory Issues**: The sentence-transformers model may require significant memory during initialization.

## Stopping the Services

To stop all services:

```bash
docker-compose down
```

To stop and remove volumes (this will delete ChromaDB data):

```bash
docker-compose down -v
```

## API Documentation

The FastAPI application automatically provides interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`