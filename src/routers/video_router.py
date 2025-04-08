import os
from enum import Enum
from fastapi import APIRouter, File, UploadFile, Query, Response, status, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List
import io
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
    description="Process a single video frame and return audio guidance directly as stream"
)
async def process_video(
    folder_name: VideoFolder = Query(..., description="Select a video folder"),
    frame_index: int = Query(..., description="Index of the frame to process"),
    tts_engine: TTSEngine = Query(..., description="Text-to-speech engine to use")
):
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
        logger.info(f"Processing folder: {folder_name} with frame index: {frame_index}, using TTS engine: {tts_engine}")

        # Process single frame and get audio response
        result = await video_handler.process_single_frame(folder_name, frame_index, tts_engine)
        
        # Check for error
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Verify audio data is available
        if not result.audio_data:
            logger.error("No audio data generated")
            raise HTTPException(status_code=500, detail="No audio data generated")
        
        # Create headers
        headers = {
            "X-Audio-Text": result.text,
            "X-Audio-Duration": str(result.duration) if result.duration else "",
            "X-Audio-Voice": result.voice if result.voice else "",
            "X-Audio-Engine": result.engine if result.engine else "",
            "X-Audio-Format": result.format if result.format else "wav",
        }
        
        # Return StreamingResponse with audio bytes
        return StreamingResponse(
            io.BytesIO(result.audio_data),
            media_type=f"audio/{result.format}",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
