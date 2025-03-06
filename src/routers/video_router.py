from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
import os
import time

from src.handlers.video_handler import VideoHandler
from src.handlers.object_detection_handler import ObjectDetectionHandler
from src.handlers.depth_estimation_handler import DepthEstimationHandler
from src.handlers.video_handler import VideoHandler
from src.utils.logger import logger

# Khởi tạo router
router = APIRouter(
    prefix="/api/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Cấu hình
OUTPUT_FRAME_PATH = os.getenv("OUTPUT_PATH") + "frames/"
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
TIME_INTERVAL = int(os.getenv("TIME_INTERVAL", "1"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

video_handler = VideoHandler(output_path=OUTPUT_FRAME_PATH, time_interval=TIME_INTERVAL)

@router.post("/process")
async def process_video(request: Request, file: UploadFile = File(...),):
    """
    Process a video file:
    1. Extract frames
    2. Detect objects in each frame
    3. Estimate depth for detected objects
    """
    try:
        # Truy cập các model từ app.state
        depth_model = request.app.state.depth_model
        gemini_client = request.app.state.gemini_client
        object_detector = ObjectDetectionHandler(gemini_client)
        depth_estimator = DepthEstimationHandler(depth_model)
        # Process video and extract frames
        start_time = time.time()
        video_result = await video_handler.process_video(file)
        if "error" in video_result:
            raise HTTPException(status_code=400, detail=video_result["error"])
            
        frames = video_result["frames"]
        logger.info(f"Extracted {len(frames)} frames from video")

        # Process each frame
        results = []
        # for frame in frames:
        # Detect objects
        objects = await object_detector.detect_objects(frames[0]["frame_path"])
        if objects:
            # Estimate depth for detected objects
            objects_with_depth = depth_estimator.estimate_depths(objects, frames[0]["frame_path"])
            execute_time = time.time() - start_time
            # Add to results
            results.append({
                "timestamp": frames[0]["timestamp"],
                "frame_path": frames[0]["frame_path"],
                "execute_time": execute_time,
                "objects": objects_with_depth
                })

        return {
            "status": "success",
            "total_frames": len(frames),
            "processed_frames": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))