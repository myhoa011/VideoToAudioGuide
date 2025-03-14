from pathlib import Path
from fastapi import UploadFile
import aiofiles
import os
from datetime import datetime

from src.utils.logger import logger
from src.helpers.video_helper import (
    extract_frames,
    validate_extension,
    cleanup_file
)
from src.handlers.object_detection_handler import ObjectDetectionHandler
from src.handlers.depth_estimation_handler import DepthEstimationHandler
from src.handlers.navigation_guide_handler import NavigationGuideHandler 
from src.utils.constant import OUTPUT_FRAME_PATH   

object_detector = ObjectDetectionHandler()
depth_estimator = DepthEstimationHandler()
navigation_guide = NavigationGuideHandler() 

class VideoHandler:
    """Handler for video processing operations"""
    
    def __init__(self, output_path: str, time_interval: int):
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
    
    @staticmethod
    def get_video_folders():
        """Get list of video folders for enum"""
        frames_base_path = os.path.abspath(OUTPUT_FRAME_PATH)
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
    
    async def process_video(self, folder_name: str, num_frames: int):
        """
        Process frames from an existing folder
        
        Args:
            folder_name (str): Name of the folder containing frames
            num_frames (int): Number of frames to process
            
        Returns:
            dict: Processing results with objects, depth and navigation data
        """
        try:
            if folder_name == "no_videos_available":
                return {"error": "No videos available. Please upload a video first."}
                
            frames_folder = os.path.join(os.path.abspath(self.output_path), folder_name)
            
            if not os.path.exists(frames_folder):
                return {"error": f"Folder '{folder_name}' not found"}

            # Get total frames in folder
            frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_')])
            total_frames = len(frame_files)
            
            frame_indices = list(range(min(num_frames, total_frames)))  
            
            # Process selected frames
            total_start_time = datetime.now()
            
            results = []
            for frame_idx in frame_indices:
                frame_path = os.path.join(frames_folder, frame_files[frame_idx])
                
                frame_result = {
                    "frame_index": frame_idx,
                    "frame_path": frame_path,
                    "objects": [],
                    "navigation": None,
                    "execution_times": {
                        "object_detection": 0,
                        "depth_estimation": 0,
                        "navigation_generation": 0,
                        "total": 0
                    }
                }
                
                # Measure object detection time
                obj_detection_start = datetime.now()
                objects = await object_detector.detect_objects(frame_path)
                obj_detection_time = (datetime.now() - obj_detection_start).total_seconds()
                frame_result["execution_times"]["object_detection"] = obj_detection_time
                
                if objects:
                    # Measure depth estimation time
                    depth_start = datetime.now()
                    objects_with_depth = depth_estimator.estimate_depths(
                        objects, 
                        frame_path
                    )
                    depth_time = (datetime.now() - depth_start).total_seconds()
                    frame_result["execution_times"]["depth_estimation"] = depth_time
                    
                    # Measure navigation guidance generation time
                    navigation_start = datetime.now()
                    navigation = await navigation_guide.generate_navigation_guide(objects_with_depth)
                    navigation_time = (datetime.now() - navigation_start).total_seconds()
                    frame_result["execution_times"]["navigation_generation"] = navigation_time
                    
                    frame_result["objects"] = objects_with_depth
                    frame_result["navigation"] = navigation
                
                # Calculate total frame processing time
                frame_result["execution_times"]["total"] = sum(frame_result["execution_times"].values())
                
                results.append(frame_result)

            total_execution_time = (datetime.now() - total_start_time).total_seconds()

            return {
                "frames_folder": frames_folder,
                "total_frames": total_frames,
                "processed_frames": len(results),
                "execution_time": total_execution_time,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return {"error": str(e)}