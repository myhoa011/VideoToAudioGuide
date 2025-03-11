import os
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from src.handlers.video_handler import VideoHandler
from src.handlers.object_detection_handler import ObjectDetectionHandler
from src.handlers.depth_estimation_handler import DepthEstimationHandler
from src.utils.logger import logger
from src.utils.constant import OUTPUT_FRAME_PATH, TIME_INTERVAL


# Initialize router
router = APIRouter(
    prefix="/api/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Initialize video handler
video_handler = VideoHandler(output_path=OUTPUT_FRAME_PATH, time_interval=TIME_INTERVAL)

@router.post("/upload")
async def save_video(file: UploadFile = File(...)):
    """
    Process a video file:
    1. Save video with timestamp
    2. Extract frames
    3. Detect objects in each frame
    4. Estimate depth for detected objects
    
    Args:
        file: Uploaded video file
        
    Returns:
        dict: Processing results including detected objects with depth
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
        dict: Processing results including detected objects with depth
    """
    try:
        # Initialize handlers
        object_detector = ObjectDetectionHandler()
        depth_estimator = DepthEstimationHandler()

        # Get total frames in folder
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.startswith('frame_')])
        total_frames = len(frame_files)
        
        # Calculate frame indices to process
        step = max(1, total_frames // num_frames)
        frame_indices = list(range(0, total_frames, step))[:num_frames]
        
        # Process selected frames
        start_time = datetime.now()
        
        results = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(frames_folder, frame_files[frame_idx])
            
            # Detect objects
            objects = await object_detector.detect_objects(frame_path)
            if objects:
                objects_with_depth = depth_estimator.estimate_depths(
                    objects, 
                    frame_path
                )
                results.append({
                    "frame_index": frame_idx,
                    "frame_path": frame_path,
                    "objects": objects_with_depth
                })

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success", 
            "frames_folder": frames_folder,
            "total_frames": total_frames,
            "processed_frames": len(results),
            "execution_time": execution_time,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
