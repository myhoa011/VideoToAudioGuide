import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(".")

from src.utils.logger import logger
from src.routers import video_router
from src.utils.constant import DEFAULT_HOST, DEFAULT_PORT


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage application lifecycle"""
    try:
        logger.info("Starting Video Analysis API")
        yield
    finally:
        logger.info("Shutting down Video Analysis API")

# Initialize FastAPI app
app = FastAPI(
    title="Video Analysis API",
    description="API for video processing, object detection and depth estimation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.app:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=True
    )
