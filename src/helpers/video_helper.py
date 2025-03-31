import cv2
from datetime import timedelta, datetime
from pathlib import Path
import asyncio
import os
from src.utils.logger import logger
from src.utils.constant import ALLOWED_EXTENSIONS
from schemas import VideoFrame

def extract_frames(video_path, output_path, time_interval) -> list:
    """
    Extract frames from the video at specified intervals and save them in 
    a subfolder named after the video filename and timestamp.
    
    Returns:
        list: List of dictionaries containing frame information, or None if an error occurs.
    """
    try:
        frames_data = []

        # Open video file using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video file")

        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a unique subfolder inside output_path for this video
        video_filename = Path(video_path).stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get timestamp
        frames_output_folder = Path(output_path) / f"{video_filename}_{timestamp}"
        frames_output_folder.mkdir(parents=True, exist_ok=True)  # Create folder if needed

        # Calculate frame interval based on FPS and time interval
        frame_interval = int(fps * time_interval)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at specified intervals
            if frame_count % frame_interval == 0:
                # Calculate current timestamp
                current_time = frame_count / fps
                video_timestamp = str(timedelta(seconds=int(current_time)))

                # Preprocess frame
                processed_frame = preprocess_frame(frame)

                # Create filename and save frame inside the subfolder
                frame_filename = f"frame_{video_timestamp.replace(':', '_')}.jpg"
                frame_path = frames_output_folder / frame_filename
                cv2.imwrite(str(frame_path), processed_frame)

                # Store frame information
                frames_data.append(VideoFrame(
                    timestamp=video_timestamp,
                    video_path=str(frame_path)
                ))

            frame_count += 1

        cap.release()

        return frames_data

    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return None
    
def preprocess_frame(frame):
    """
    Preprocess the extracted frame to optimize quality.
    
    Processing steps:
    1. Resize to 720p resolution
    2. Adjust brightness and contrast
    3. Apply noise reduction
    
    Args:
        frame (numpy.ndarray): Input frame to process
        
    Returns:
        numpy.ndarray: Processed frame, or original frame if processing fails
    """
    try:
        # Resize to 720p
        frame = cv2.resize(frame, (1280, 720))
        
        # Apply custom processing based on frame content
        frame_mean = frame.mean()
        if frame_mean < 50:
            # Dark frame, increase brightness more
            alpha = 1.5
            beta = 20
        elif frame_mean > 200:
            # Bright frame, decrease brightness
            alpha = 1.0
            beta = -10
        else:
            # Normal frame, use default values
            alpha = 1.2
            beta = 10
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Reduce noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        return frame
        
    except Exception as e:
        print(f"Error preprocessing frame: {str(e)}")
        return frame 

def validate_extension(video_path: str, allowed_extensions=ALLOWED_EXTENSIONS) -> bool:
    """
    Validate if the video file has an allowed extension.
    
    Args:
        video_path (str): Path to the video file.
        allowed_extensions (set): Set of allowed file extensions.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return Path(video_path).suffix.lower() in allowed_extensions

def cleanup_video(video_path: str) -> bool:
    """
    Delete video file if exists.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        bool: True if deleted or didn't exist, False if error
    """
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Cleaned up video: {video_path}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning up video: {str(e)}")
        return False
    
async def cleanup_file(file_path: str) -> bool:
    """
    Delete a file from the system.

    Args:
        file_path (str): Path to the file to delete.

    Returns:
        bool: True if file was deleted successfully, False otherwise.
    """
    try:
        file = Path(file_path)
        if file.exists():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, file.unlink)  # Non-blocking execution
            logger.info(f"Deleted file: {file_path}")
            return True
        else:
            logger.warning(f"File not found: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        return False

def parse_frame_range(num_frames_str: str) -> tuple:
    """
    Parse a frame range string into start and end frame indices
    
    Args:
        num_frames_str (str): String representation of frame range (e.g., "5" or "3,7")
    
    Returns:
        tuple: (start_frame, end_frame) or raises ValueError if invalid
    """
    frame_parts = num_frames_str.strip().split(',')
    if len(frame_parts) == 1:
        # Single number - process from 0 to that number
        start_frame = 0
        end_frame = int(frame_parts[0])
    elif len(frame_parts) == 2:
        # Range - process from first to second number
        start_frame = int(frame_parts[0])
        end_frame = int(frame_parts[1])
    else:
        raise ValueError("Invalid frame range format")
        
    # Validate range
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid frame range values")
        
    return start_frame, end_frame
    
def get_video_folders(frames_base_path: str) -> dict:
    """
    Get list of video folders from the frames directory
    
    Args:
        frames_base_path (str): Base path where frame folders are stored
        
    Returns:
        dict: Dictionary of folder names
    """
    folders = {}
    
    if os.path.exists(frames_base_path):
        for item in os.listdir(frames_base_path):
            item_path = os.path.join(frames_base_path, item)
            if os.path.isdir(item_path):
                frame_files = [f for f in os.listdir(item_path) if f.startswith('frame_')]
                if frame_files:
                    folders[item] = item
    
    # If no folders found, add a placeholder
    if not folders:
        folders["no_videos_available"] = "no_videos_available"
        
    return folders