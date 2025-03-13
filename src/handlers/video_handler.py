from pathlib import Path
from fastapi import UploadFile
import aiofiles

from src.utils.logger import logger
from src.helpers.video_helper import (
    extract_frames,
    validate_extension,
    cleanup_file
)


class VideoHandler:
    """Handler for video processing operations"""
    
    def __init__(self, output_path: str, time_interval):
        self.output_path = Path(output_path)
        self.time_interval = time_interval

    async def extract_frames(self, video_file: UploadFile) -> dict:
        """
        Process uploaded video file
        
        Args:
            video_file (UploadFile): Uploaded video file
            
        Returns:
            dict: Processing results including frame analysis
        """
        temp_path = None
        try:
            # Validate file extension
            if not validate_extension(video_file.filename):
                logger.error(f"Invalid file extension: {video_file.filename}")
                return {"error": "Invalid file type"}
                
            # Save uploaded file
            temp_path = self.output_path / video_file.filename
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await video_file.read()
                await out_file.write(content)
            
            frames_data = extract_frames(temp_path, self.output_path, time_interval=self.time_interval)
            if not frames_data:
                return {"error": "Failed to process video"}
            
            # Cleanup video file after successful processing
            await cleanup_file(str(temp_path))
                
            return {
                "status": "success",
                "frames": frames_data
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            # Try to cleanup even if processing failed
            if temp_path:
                await cleanup_file(temp_path)
            return {"error": str(e)}