import cv2
from datetime import datetime, timedelta
from pathlib import Path
import json

class VideoProcessor:
    """
    A class for processing video files and extracting frames at specified intervals.
    
    This class handles:
    - Video frame extraction at regular time intervals
    - Frame preprocessing (resizing, enhancing, noise reduction)
    - Organized storage of frames in timestamped directories
    - Metadata generation for extracted frames
    
    Attributes:
        video_path (str): Path to the input video file
        output_base_path (str): Base directory for storing extracted frames
        time_interval (int): Time interval between extracted frames in seconds
        frames_data (list): List of dictionaries containing frame information
        output_path (str): Full path to the directory where frames will be stored
    """

    def __init__(self, video_path: str, output_path: str, time_interval: int = 1):
        """
        Initialize the VideoProcessing object.

        Args:
            video_path (str): Path to the input video file
            output_path (str): Base directory for storing extracted frames
            time_interval (int, optional): Time interval between frames in seconds. Defaults to 1.
        """
        self.video_path = Path(video_path)
        self.output_base_path = Path(output_path)
        self.time_interval = time_interval
        self.frames_data = []
        
        # Create unique directory name for this video processing
        video_name = self.video_path.stem
        self.output_path = self.output_base_path / f"{video_name}"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def extract_frames(self) -> list:
        """
        Extract frames from the video at specified intervals.
        
        The method:
        1. Opens the video file
        2. Calculates frame extraction points based on FPS and interval
        3. Extracts and preprocesses frames
        4. Saves frames as JPG files
        5. Generates metadata for the extraction process
        
        Returns:
            list: List of dictionaries containing frame information, or None if an error occurs.
            Each dictionary contains:
                - timestamp: Time position in the video
                - frame_path: Path to the saved frame image
        """
        try:
            # Open video file using OpenCV
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file")

            # Get video information
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Calculate frame interval based on FPS and time interval
            frame_interval = int(fps * self.time_interval)
            
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
                    processed_frame = self.preprocess_frame(frame)
                    
                    # Create filename and save frame
                    frame_filename = f"frame_{video_timestamp.replace(':', '_')}.jpg"
                    frame_path = self.output_path / frame_filename
                    cv2.imwrite(str(frame_path), processed_frame)
                    
                    # Store frame information
                    self.frames_data.append({
                        'timestamp': video_timestamp,
                        'frame_path': str(frame_path)
                    })
                    
                frame_count += 1
                
            cap.release()
            
            # Save metadata for extracted frames
            self._save_metadata()
            
            return self.frames_data
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return None

    def preprocess_frame(self, frame):
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
            return frame  # Return original frame if processing fails

    def _save_metadata(self) -> None:
        """
        Save metadata about the extracted frames to a JSON file.
        
        The metadata includes:
        - Original video path
        - Extraction timestamp
        - List of extracted frames with their timestamps and paths
        """
        metadata_path = self.output_path / "metadata.json"
        metadata = {
            "video_path": str(self.video_path),
            "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": len(self.frames_data),
            "frames": self.frames_data
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False) 