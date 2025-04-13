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
    cleanup_file,
    parse_frame_range,
    get_video_folders
)
from src.handlers.object_detection_handler import ObjectDetectionHandler
from src.handlers.depth_estimation_handler import DepthEstimationHandler
from src.handlers.navigation_guide_handler import NavigationGuideHandler
from src.utils.constant import OUTPUT_FRAME_PATH, CONCURRENCY_LIMIT
from src.handlers.text_to_speech_handler import TextToSpeechHandler
from src.helpers.report_helper import save_execution_time_to_csv, save_video_analysis_to_csv
from src.schemas.navigation import NavigationGuide
from src.schemas import ExecutionTime, FrameAnalysis, VideoProcessingResult, AudioResponse, VideoAnalysisResponse

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
            
            logger.info(f"Processing video: {video_file.filename}")

            # Save uploaded file
            temp_path = self.output_path / video_file.filename
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await video_file.read()
                await out_file.write(content)
            
            frames_data = extract_frames(temp_path, self.output_path, time_interval=self.time_interval)
            if not frames_data:
                raise Exception("Failed to process video")
            
            logger.info(f"Extracted {len(frames_data)} frames from video")
            
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
        return get_video_folders(frames_base_path)
    
    async def process_frame(self, folder_name: str, frame_idx: int, frame_path: str, tts_engine: str) -> FrameAnalysis:
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
        else:
            # Create default NavigationGuide object if no objects detected
            navigation_guide_obj = NavigationGuide(
                navigation_text="No objects detected, the path ahead is clear.",
                priority_objects=[]
            )
            execution_time.navigation_generation = 0.0
        
        # Perform text-to-speech
        tts_start = datetime.now()
        audio_data = await tts_handler.convert_text_to_speech(
            navigation_guide_obj.navigation_text,
            folder_name,
            str(frame_idx),
            tts_engine
        )
        tts_time = (datetime.now() - tts_start).total_seconds()
        execution_time.text_to_speech = tts_time
        
        # Calculate total processing time
        execution_time.total = sum([
            execution_time.object_detection,
            execution_time.depth_estimation or 0.0,
            execution_time.navigation_generation or 0.0,
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

        logger.debug(f"Frame {frame_idx} execution time: {execution_time}")
        
        return frame_analysis
    
    async def process_frames_string(self, folder_name: str, num_frames_str: str, tts_engine: str) -> List[AudioResponse]:
        """
        Process video frames by parsing a frame range string
        
        Args:
            folder_name: Folder name containing frames
            num_frames_str: String representation of frame range (e.g., "5" or "3,7")
            tts_engine: Text-to-speech engine to use
            
        Returns:
            List[AudioResponse]: List of audio responses for each processed frame
        """
        try:
            # Parse frame range
            try:
                start_frame, end_frame = parse_frame_range(num_frames_str)
            except ValueError as e:
                logger.error(f"Invalid frame range: {str(e)}")
                return {"error": f"Invalid frame range: {str(e)}. Format should be a single number (e.g., '5') or a range (e.g., '3,7')."}
            
            logger.info(f"Parsed frame range: {start_frame} to {end_frame}")
            
            # Call the actual processing method
            return await self.process_frames_range(folder_name, start_frame, end_frame, tts_engine)
            
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return {"error": str(e)}
    
    async def process_frames_range(self, folder_name: str, start_frame: int, end_frame: int, tts_engine: str) -> List[AudioResponse]:
        """
        Process video frames within a specific range with parallel processing
        and return audio responses
        
        Args:
            folder_name: Folder name containing frames
            start_frame: Index of the first frame to process
            end_frame: Index of the last frame to process
            tts_engine: Text-to-speech engine to use
            
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
            
            # Validate frame range
            if start_frame >= total_frames or end_frame >= total_frames:
                logger.error(f"Invalid frame range: {start_frame} to {end_frame}. Total frames: {total_frames}")
                return {"error": f"Invalid frame range: {start_frame} to {end_frame}. Total frames: {total_frames}"}
            
            # Select frames to process
            frame_indices = list(range(start_frame, min(end_frame + 1, total_frames)))
            
            # Process frames in parallel with concurrency control
            total_start_time = datetime.now()
            
            # Limit concurrency to avoid overwhelming system resources
            concurrency_limit = CONCURRENCY_LIMIT
            semaphore = Semaphore(concurrency_limit)
            
            # Collection for execution times
            execution_times = []
            
            frames_analysis = []

            async def process_with_semaphore(idx):
                try:
                    async with semaphore:
                        frame_path = os.path.join(frames_folder, frame_files[idx])
                        # Process frame and get full analysis
                        frame_analysis = await self.process_frame(folder_name, idx, frame_path, tts_engine)
                        
                        if frame_analysis is None or frame_analysis.audio is None:
                            logger.warning(f"Frame {idx} processing returned None")
                            return None
                            
                        # Store execution time
                        execution_times.append(frame_analysis.execution_time)
                        frames_analysis.append(frame_analysis)
                        
                        # Log the detailed analysis
                        logger.info(f"Frame {idx} analysis: Objects detected: {len(frame_analysis.objects)}")
                        logger.info(f"Frame {idx} navigation: {frame_analysis.navigation.navigation_text}")
                        
                        return frame_analysis.audio
                except Exception as e:
                    logger.error(f"Error processing frame {idx}: {str(e)}")
                    return None

            # Process frames in parallel
            tasks = [process_with_semaphore(idx) for idx in frame_indices]
            audio_responses = await gather(*tasks)
            
            # Count successful vs failed frames
            successful_frames = len([r for r in audio_responses if r is not None])
            failed_frames = len(audio_responses) - successful_frames
            logger.info(f"Processed {successful_frames} frames successfully, {failed_frames} frames failed")
            
            if successful_frames == 0:
                raise Exception("All frames failed to process")

            total_execution_time = (datetime.now() - total_start_time).total_seconds()
            logger.info(f"Total processing time: {total_execution_time:.2f} seconds for frames {start_frame} to {end_frame}")
            
            # Filter out None values (in case some frames didn't generate audio)
            audio_responses = [audio for audio in audio_responses if audio is not None]

            video_response = VideoAnalysisResponse(
                video_path=frames_folder,
                total_frames=total_frames,
                frames_analysis=frames_analysis,
            )

            # # Save video analysis report to CSV
            # report_path = save_video_analysis_to_csv(frames_analysis, folder_name)
            # logger.info(f"Video analysis report saved to: {report_path}")
            report_path = save_execution_time_to_csv(execution_times, folder_name)
            logger.info(f"Execution time report saved to: {report_path}")
            return audio_responses
    
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return {"error": str(e)}

    async def process_single_frame(self, folder_name: str, frame_index: int, tts_engine: str) -> AudioResponse:
        """
        Process a single video frame and return audio response
        
        Args:
            folder_name: Folder name containing frames
            frame_index: Index of the frame to process
            tts_engine: Text-to-speech engine to use
            
        Returns:
            AudioResponse: Audio response with audio_data
        """
        try:
            if folder_name == "no_videos_available":
                logger.error("No videos available. Please upload a video first.")
                return {"error": "No videos available. Please upload a video first."}
                
            frames_folder = os.path.join(os.path.abspath(self.output_path), folder_name)
            
            if not os.path.exists(frames_folder):
                logger.error(f"Folder '{folder_name}' not found")
                return {"error": f"Folder '{folder_name}' not found"}
    
            # Get frames in folder
            frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_')])
            total_frames = len(frame_files)
            
            # Validate frame index
            if frame_index < 0 or frame_index >= total_frames:
                logger.error(f"Invalid frame index: {frame_index}. Total frames: {total_frames}")
                return {"error": f"Invalid frame index: {frame_index}. Total frames: {total_frames}"}
            
            # Get frame path
            frame_path = os.path.join(frames_folder, frame_files[frame_index])
            
            # Process the frame
            logger.info(f"Processing single frame at index {frame_index}: {frame_path}")
            frame_analysis = await self.process_frame(folder_name, frame_index, frame_path, tts_engine)
            
            if frame_analysis is None or frame_analysis.audio is None:
                logger.error(f"Frame {frame_index} processing failed")
                return {"error": f"Frame {frame_index} processing failed"}
            
            # Get the audio response
            audio_response = frame_analysis.audio
            
            # Skip file existence check since we're streaming directly
            return audio_response
            
        except Exception as e:
            logger.error(f"Error processing single frame: {str(e)}")
            return {"error": str(e)}