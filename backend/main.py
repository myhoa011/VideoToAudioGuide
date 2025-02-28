from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import time
from datetime import datetime
import aiofiles
from pathlib import Path
from werkzeug.utils import secure_filename
import logging
from typing import Optional

from module.video_processor.video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Frame Extraction API",
    description="API for extracting and processing frames from video files",
    version="1.0.0"
)

# Configuration
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "frames/"))
TIME_INTERVAL = int(os.getenv("TIME_INTERVAL", "1"))
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def validate_video_file(filename: str) -> bool:
    """
    Validate if the uploaded file is a video with allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload and process a video file.
    
    Args:
        file (UploadFile): The video file to be processed
        
    Returns:
        JSONResponse: Processing results including frame data and execution time
        
    Raises:
        HTTPException: If file validation or processing fails
    """
    start_time = time.time()
    temp_file_path: Optional[Path] = None
    
    try:
        # Validate file extension
        if not validate_video_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Secure filename and create temporary path
        filename = secure_filename(file.filename)
        formatted_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file_path = OUTPUT_PATH / f'{Path(filename).stem}_{formatted_time}{Path(filename).suffix}'

        # Save uploaded file
        logger.info(f"Saving uploaded file to {temp_file_path}")
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Process video
        logger.info("Starting video processing")
        result = await process_video(str(temp_file_path))
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Video processing failed"
            )

        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.2f} seconds")

        return JSONResponse(content={
            "message": "Video processing completed successfully",
            "frames_extracted": len(result),
            "output_directory": str(result[0]['frame_path'] if result else ''),
            "execution_time": f"{total_time:.2f} seconds",
            "frames_data": result
        })

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An error occurred while processing the video",
                "error": str(e)
            }
        )

    finally:
        # Cleanup temporary file
        if temp_file_path and temp_file_path.exists():
            logger.info(f"Cleaning up temporary file: {temp_file_path}")
            temp_file_path.unlink()

async def process_video(video_path: str) -> list:
    """
    Process the video and extract frames.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        list: List of dictionaries containing frame information
    """
    try:
        video_processor = VideoProcessor(
            video_path=video_path,
            output_path=str(OUTPUT_PATH),
            time_interval=TIME_INTERVAL
        )
        return video_processor.extract_frames()
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 