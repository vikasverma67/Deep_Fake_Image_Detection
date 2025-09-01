# Deep-Detect Backend

This is the FastAPI backend for the Deep-Detect application, providing deepfake detection and image metadata analysis capabilities.

## Setup

1. Create and activate virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Start the development server:

```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Deepfake Detection

- **Endpoint**: `/api/detect`
- **Method**: POST
- **Input**: Image file
- **Response**:
  ```json
  {
    "prediction": "real/fake",
    "confidence": 0.95,
    "metadata": {
      // Detailed metadata analysis
    }
  }
  ```

### 2. Metadata Analysis

- **Endpoint**: `/api/metadata`
- **Method**: POST
- **Input**: Image file
- **Response**: Detailed image metadata including:
  - File information
  - Image properties
  - EXIF data
  - Compression analysis

## API Documentation

Once the server is running, you can access:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── services/
│   ├── model_service.py    # Deepfake detection service
│   └── metadata_service.py # Image metadata analysis service
└── venv/               # Virtual environment
```

## Dependencies

- FastAPI: Web framework
- ONNX Runtime: Model inference
- Pillow: Image processing
- python-magic: File type detection
- exif: EXIF data extraction
- uvicorn: ASGI server
