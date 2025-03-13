import os
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from src.handlers.video_handler import VideoHandler
from src.handlers.object_detection_handler import ObjectDetectionHandler
from src.handlers.depth_estimation_handler import DepthEstimationHandler
from src.handlers.navigation_guide_handler import NavigationGuideHandler
from src.utils.logger import logger
from src.utils.constant import OUTPUT_FRAME_PATH, TIME_INTERVAL

# Initialize router
router = APIRouter(
    prefix="/api/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Initialize handler
video_handler = VideoHandler(output_path=OUTPUT_FRAME_PATH, time_interval=TIME_INTERVAL)
object_detector = ObjectDetectionHandler()
depth_estimator = DepthEstimationHandler()
navigation_guide = NavigationGuideHandler()

@router.post("/upload")
async def save_video(file: UploadFile = File(...)):
    """
    Process a video file:
    1. Save video with timestamp
    2. Extract frames
    
    Args:
        file: Uploaded video file
        
    Returns:
        dict: Save results
    """
    try:
        # Process video and extract frames
        video_result = await video_handler.extract_frames(file)
        if "error" in video_result:
            raise HTTPException(status_code=400, detail=video_result["error"])
            
        frames = video_result["frames"]
        video_path = video_handler.output_path
        
        logger.info(f"Extracted {len(frames)} frames from video: {video_path}")

        return {
            "status": "success",
            "video_path": video_path,
            "total_frames": len(frames)
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/process")
async def process_video(
    frames_folder: str = Form(...),
    num_frames: int = Form(...)
):
    """
    Process selected frames from uploaded video
    
    Args:
        frames_folder: Path to folder containing frames
        num_frames: Number of frames to process
        
    Returns:
        dict: Processing results including detected objects with depth and navigation guidance
              with execution time for each processing step
    """
    try:
        if not os.path.isabs(frames_folder):
            frames_folder = os.path.join(os.path.abspath(os.getcwd()), "outputs", "frames", frames_folder)

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
            "status": "success", 
            "frames_folder": frames_folder,
            "total_frames": total_frames,
            "processed_frames": len(results),
            "execution_time": total_execution_time,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
