from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
try:
    from services.model_service import ModelService
    from services.metadata_service import MetadataService
except ImportError as e:
    logger.error(f"Error importing services: {e}")
    raise

app = FastAPI(
    title="Deep-Detect API",
    description="API for deepfake detection and image metadata analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    model_service = ModelService()
    metadata_service = MetadataService()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    raise

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    metadata: Dict[str, Any]

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Endpoint for deepfake detection using the ML model
    """
    try:
        # Save the uploaded file temporarily
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get model prediction
        prediction, confidence = model_service.predict(temp_path)
        
        # Get metadata analysis
        metadata = metadata_service.analyze(temp_path)
        
        # Clean up temporary file
        temp_path.unlink()
        
        return DetectionResponse(
            prediction=prediction,
            confidence=confidence,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error in detect_deepfake: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metadata")
async def analyze_metadata(file: UploadFile = File(...)):
    """
    Endpoint for detailed image metadata analysis
    """
    try:
        # Save the uploaded file temporarily
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get metadata analysis
        metadata = metadata_service.analyze(temp_path)
        
        # Clean up temporary file
        temp_path.unlink()
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error in analyze_metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise 