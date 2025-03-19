from pathlib import Path
from fastapi import UploadFile
import aiofiles
import os
from datetime import datetime
from asyncio import gather, Semaphore
from typing import List

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
from src.handlers.text_to_speech_handler import TextToSpeechHandler
from src.schemas.navigation import NavigationGuide
from schemas import ExecutionTime, FrameAnalysis, VideoProcessingResult, AudioResponse

object_detector = ObjectDetectionHandler()
depth_estimator = DepthEstimationHandler()
navigation_guide = NavigationGuideHandler()
tts_handler = TextToSpeechHandler()

class VideoHandler:
    """Handler for video processing operations"""
    
    def __init__(self, output_path: str, time_interval: int):
        self.output_path = Path(output_path)
        self.time_interval = time_interval

    async def extract_frames(self, video_file: UploadFile) -> VideoProcessingResult:
        """
        Process uploaded video file
        
        Args:
            video_file (UploadFile): Uploaded video file
            
        Returns:
            VideoProcessingResult: Processing results including frame analysis
        """
        temp_path = None
        try:
            # Validate file extension
            if not validate_extension(video_file.filename):
                logger.error(f"Invalid file extension: {video_file.filename}")
                raise Exception("Invalid file type")
                
            # Save uploaded file
            temp_path = self.output_path / video_file.filename
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await video_file.read()
                await out_file.write(content)
            
            frames_data = extract_frames(temp_path, self.output_path, time_interval=self.time_interval)
            if not frames_data:
                raise Exception("Failed to process video")
            
            # Cleanup video file after successful processing
            await cleanup_file(str(temp_path))
                
            result = VideoProcessingResult(
                status="success",
                video_path=str(temp_path),
                total_frames=len(frames_data),
                frames=frames_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            # Try to cleanup even if processing failed
            if temp_path:
                await cleanup_file(temp_path)
            raise Exception(str(e))
    
    @staticmethod
    def get_video_folders() -> dict:
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
    
    async def process_frame(self, folder_name: str, frame_idx: int, frame_path: str) -> FrameAnalysis:
        """
        Process a single frame asynchronously
        
        Args:
            folder_name: Folder name containing frames
            frame_idx: Index of the frame to process
            frame_path: Path to the frame file
            
        Returns:
            FrameAnalysis: Analysis results for the frame
        """
        logger.info(f"Processing frame: {os.path.basename(frame_path)} (index {frame_idx})")
        
        # Measure object detection time
        obj_detection_start = datetime.now()
        objects = await object_detector.detect_objects(frame_path)
        obj_detection_time = (datetime.now() - obj_detection_start).total_seconds()
        
        # Create ExecutionTime object
        execution_time = ExecutionTime(
            object_detection=obj_detection_time
        )
        
        objects_with_depth = []
        navigation_guide_obj = None
        audio_data = None
        
        if objects:
            # Measure depth estimation time
            depth_start = datetime.now()
            objects_with_depth = depth_estimator.estimate_depths(objects, frame_path)
            depth_time = (datetime.now() - depth_start).total_seconds()
            execution_time.depth_estimation = depth_time
            
            # Measure navigation guidance generation time
            navigation_start = datetime.now()
            navigation_guide_obj = await navigation_guide.generate_navigation_guide(objects_with_depth)
            navigation_time = (datetime.now() - navigation_start).total_seconds()
            execution_time.navigation_generation = navigation_time
            
            # Perform text-to-speech
            tts_start = datetime.now()
            audio_data = await tts_handler.convert_text_to_speech(
                navigation_guide_obj.navigation_text,
                folder_name,
                str(frame_idx)
            )
            tts_time = (datetime.now() - tts_start).total_seconds()
            execution_time.text_to_speech = tts_time
        else:
            # Create default NavigationGuide object if no objects detected
            navigation_guide_obj = NavigationGuide(
                navigation_text="No objects detected, the path ahead is clear.",
                priority_objects=[]
            )
        
        # Calculate total processing time
        execution_time.total = sum([
            execution_time.object_detection,
            execution_time.depth_estimation,
            execution_time.navigation_generation,
            execution_time.text_to_speech
        ])
        
        # Create FrameAnalysis object
        frame_analysis = FrameAnalysis(
            frame_index=str(frame_idx),
            frame_path=frame_path,
            objects=objects_with_depth,
            navigation=navigation_guide_obj.model_dump(),
            audio=audio_data.model_dump(),
            execution_time=execution_time
        )
        
        return frame_analysis
    
    async def process_video(self, folder_name: str, num_frames: int) -> List[AudioResponse]:
        """
        Process video from a folder containing frames with parallel processing
        but only return audio responses
        
        Args:
            folder_name: Folder name containing frames
            num_frames: Number of frames to process
            
        Returns:
            List[AudioResponse]: List of audio responses for each processed frame
        """
        try:
            if folder_name == "no_videos_available":
                logger.error("No videos available. Please upload a video first.")
                return {"error": "No videos available. Please upload a video first."}
                
            frames_folder = os.path.join(os.path.abspath(self.output_path), folder_name)
            
            if not os.path.exists(frames_folder):
                logger.error(f"Folder '{folder_name}' not found")
                return {"error": f"Folder '{folder_name}' not found"}
    
            # Get total frames in folder
            frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_')])
            total_frames = len(frame_files)
            
            # Select frames to process
            frame_indices = list(range(min(num_frames, total_frames)))
            
            # Process frames in parallel with concurrency control
            total_start_time = datetime.now()
            
            # Limit concurrency to avoid overwhelming system resources
            concurrency_limit = 5
            semaphore = Semaphore(concurrency_limit)
            
            async def process_with_semaphore(idx):
                async with semaphore:
                    frame_path = os.path.join(frames_folder, frame_files[idx])
                    # Process frame and get full analysis
                    frame_analysis = await self.process_frame(folder_name, idx, frame_path)
                    
                    # Log the detailed analysis instead of returning it
                    logger.info(f"Frame {idx} analysis: Objects detected: {len(frame_analysis.objects)}")
                    logger.info(f"Frame {idx} navigation: {frame_analysis.navigation.navigation_text}")
                    
                    # Only return the audio data
                    return frame_analysis.audio
            
            # Process frames in parallel
            tasks = [process_with_semaphore(idx) for idx in frame_indices]
            audio_responses = await gather(*tasks)
            
            total_execution_time = (datetime.now() - total_start_time).total_seconds()
            logger.info(f"Total processing time: {total_execution_time:.2f} seconds")
            
            # Filter out None values (in case some frames didn't generate audio)
            audio_responses = [audio for audio in audio_responses if audio is not None]
            
            return audio_responses
    
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return {"error": str(e)}