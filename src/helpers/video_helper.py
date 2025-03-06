import cv2
from datetime import timedelta, datetime
from pathlib import Path
from src.utils.logger import logger

def extract_frames(video_path, output_path, time_interval) -> list:
    """
    Extract frames from the video at specified intervals and save them in 
    a subfolder named after the video filename and timestamp.

    The method:
    1. Opens the video file
    2. Calculates frame extraction points based on FPS and interval
    3. Extracts and preprocesses frames
    4. Saves frames as JPG files inside a dedicated subfolder
    5. Generates metadata for the extraction process

    Returns:
        list: List of dictionaries containing frame information, or None if an error occurs.
        Each dictionary contains:
            - timestamp: Time position in the video
            - frame_path: Path to the saved frame image
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
        duration = total_frames / fps

        # Create a unique subfolder inside output_path for this video
        video_filename = Path(video_path).stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get timestamp
        video_output_folder = Path(output_path) / f"{video_filename}_{timestamp}"
        video_output_folder.mkdir(parents=True, exist_ok=True)  # Create folder if needed

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
                frame_path = video_output_folder / frame_filename
                cv2.imwrite(str(frame_path), processed_frame)

                # Store frame information
                frames_data.append({
                    'timestamp': video_timestamp,
                    'frame_path': str(frame_path)
                })

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

def validate_extension(video_path: str, allowed_extensions={'.mp4', '.avi', '.mov', '.mkv'}) -> bool:
    """
    Validate if the video file has an allowed extension.
    
    Args:
        video_path (str): Path to the video file.
        allowed_extensions (set): Set of allowed file extensions.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return Path(video_path).suffix.lower() in allowed_extensions

import asyncio

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
