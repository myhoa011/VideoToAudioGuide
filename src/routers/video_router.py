import os
from enum import Enum
from fastapi import APIRouter, File, UploadFile, Form, Query, Response, status, HTTPException
from typing import Optional
from src.handlers.video_handler import VideoHandler
from src.utils.logger import logger
from src.utils.constant import OUTPUT_FRAME_PATH, TIME_INTERVAL, OUTPUT_AUDIO_PATH
from schemas import VideoAnalysisResponse, VideoProcessingResult

# Initialize router
router = APIRouter(
    prefix="/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Initialize handler
video_handler = VideoHandler(output_path=OUTPUT_FRAME_PATH, time_interval=TIME_INTERVAL)

# Create VideoFolder enum dynamically
VideoFolder = Enum('VideoFolder', video_handler.get_video_folders())

@router.post(
    "/upload",
    description="Process video file: save and extract frames",
    response_model=VideoProcessingResult
)
async def upload_video(
    response: Response,
    file: UploadFile = File(...)
) -> VideoProcessingResult:
    try:
        # Process video and extract frames
        video_result = await video_handler.extract_frames(file)
        response.status_code = status.HTTP_200_OK
        return video_result
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return VideoProcessingResult(
            status="error",
            video_path="",
            total_frames=0,
            frames=[]
        )

@router.post(
    "/process",
    description="Analyze video frames with object detection, depth estimation and navigation guidance",
    response_model=VideoAnalysisResponse
)
async def process_video(
    folder_name: VideoFolder = Query(..., description="Select a video folder"),
    num_frames: int = Form(..., description="Number of frames to process")
) -> VideoAnalysisResponse:
    try:
        # Get the string value from enum
        folder_name = folder_name.value
        logger.info(f"Processing folder: {folder_name}")

        # Process video frames and generate analysis
        result = await video_handler.process_video(folder_name, num_frames)
        
        # Check for error
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
