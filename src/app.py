"""
Main FastAPI application entry point.
This module initializes and configures the FastAPI application.
"""

import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

sys.path.append(".")
from src.utils.logger import logger
from src.routers import video_router
from src.initializer import Initializer

# Khởi tạo model manager
initializer = Initializer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize and cleanup application resources"""
    try:
        logger.info("Starting Video Analysis API")
        
        # Khởi tạo models và resources
        await initializer.initialize()
        app.state.depth_model = initializer.get_depth_model()
        app.state.gemini_client = initializer.get_gemini_client()

        yield  # Chạy ứng dụng trong khoảng thời gian này

    finally:
        logger.info("Shutting down Video Analysis API")

app = FastAPI(
    title="Video Analysis API",
    description="API for processing videos, detecting objects and estimating depth",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "depth_model": initializer.depth_model is not None,
            "gemini_client": initializer.gemini_client is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
