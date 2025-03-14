import os
from enum import Enum
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from src.handlers.video_handler import VideoHandler
from src.utils.logger import logger
from src.utils.constant import OUTPUT_FRAME_PATH, TIME_INTERVAL

# Initialize router
router = APIRouter(
    prefix="/api/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Initialize handler
video_handler = VideoHandler(output_path=OUTPUT_FRAME_PATH, time_interval=TIME_INTERVAL)

# Create VideoFolder enum dynamically
VideoFolder = Enum('VideoFolder', video_handler.get_video_folders())

@router.post("/upload")
async def save_video(file: UploadFile = File(...)):
    """
    Process a video file:
    1. Save video with timestamp
    2. Extract frames
    
    Args:
        file: Uploaded video file
        
    Returns:
        dict: Save results
    """
    try:
        # Process video and extract frames
        video_result = await video_handler.extract_frames(file)
        if "error" in video_result:
            raise HTTPException(status_code=400, detail=video_result["error"])
            
        frames = video_result["frames"]
        video_path = video_handler.output_path
        
        logger.info(f"Extracted {len(frames)} frames from video: {video_path}")

        return {
            "status": "success",
            "video_path": video_path,
            "total_frames": len(frames)
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/process")
async def process_video(
    frames_folder: VideoFolder = Query(..., description="Select a video"),
    num_frames: int = Form(...)
):
    try:
        # Get the string value from enum
        folder_name = frames_folder.value
        logger.info(f"Processing folder: {folder_name}")
        
        # Call processing function from handler
        result = await video_handler.process_video(
            folder_name, 
            num_frames
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result

    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
