import os
from enum import Enum
from fastapi import APIRouter, File, UploadFile, Form, Query, Response, status, HTTPException
from typing import List
from src.handlers.video_handler import VideoHandler
from src.utils.logger import logger
from src.utils.constant import OUTPUT_FRAME_PATH, TIME_INTERVAL, TTS_ENGINES
from schemas import VideoProcessingResult, AudioResponse

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
Engine = [(engine, engine) for engine in TTS_ENGINES]
TTSEngine = Enum('TTSEngine', Engine)

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
    description="Process video frames and return only audio guidance",
    response_model=List[AudioResponse]
)
async def process_video(
    folder_name: VideoFolder = Query(..., description="Select a video folder"),
    num_frames: int = Query(..., description="Number of frames to process"),
    tts_engine: TTSEngine = Query(..., description="Text-to-speech engine to use")
) -> List[AudioResponse]:
    try:
        tts_engine = tts_engine.value
        # Validate TTS engine
        if tts_engine not in TTS_ENGINES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid TTS engine. Choose from: {', '.join(TTS_ENGINES)}"
            )
            
        # Get the string value from enum
        folder_name = folder_name.value
        logger.info(f"Processing folder: {folder_name} with TTS engine: {tts_engine}")

        # Process video frames and generate analysis with specified TTS engine
        result = await video_handler.process_video(folder_name, num_frames, tts_engine)
        
        # Check for error
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
