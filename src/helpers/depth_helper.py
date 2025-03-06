from transformers import pipeline
from PIL import Image
import numpy as np
import json
from src.utils.logger import logger

def predict(depth_model, objects, image_path: str) -> list:
    """
    Estimate depth for detected objects
    
    Args:
        objects (list): List of detected objects
        image_path (str): Path to image file
        
    Returns:
        list: Objects with depth information
    """
    try:
        # Load and process image
        image = Image.open(image_path)
        depth_map = np.array(depth_model(image)['depth'])
        
        # Get depth for each object
        results = _get_object_depths(depth_map, objects)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in depth estimation: {str(e)}")
        return []

def _get_object_depths(depth_map: np.ndarray, objects) -> list:
    """
    Calculate depth for each object using the depth map
    
    Args:
        depth_map (np.ndarray): Generated depth map
        objects (list): List of detected objects
        
    Returns:
        list: Objects with calculated depths
    """
    try:
        # Parse objects if string
        if isinstance(objects, str):
            try:
                objects = json.loads(objects)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse objects JSON: {str(e)}")
                return []

        # Get depth map dimensions
        height, width = depth_map.shape
        
        # Normalize depth map to [0,1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        results = []
        for obj in objects:
            try:
                # Get coordinates
                y1, x1, y2, x2 = obj["box_2d"]
                
                # Convert to pixels
                x1_pixel = int(x1 * width / 1000)
                y1_pixel = int(y1 * height / 1000)
                x2_pixel = int(x2 * width / 1000)
                y2_pixel = int(y2 * height / 1000)
                
                # Ensure coordinates are within image
                x1_pixel = max(0, min(x1_pixel, width-1))
                y1_pixel = max(0, min(y1_pixel, height-1))
                x2_pixel = max(0, min(x2_pixel, width-1))
                y2_pixel = max(0, min(y2_pixel, height-1))
                
                # Calculate depth
                depth_region = depth_map[y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                depth_mean = float(np.mean(depth_region))
                
                # Add depth to object
                result = obj.copy()
                result["depth"] = depth_mean
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing object {obj.get('label', 'unknown')}: {str(e)}")
                result = obj.copy()
                result["depth"] = np.nan
                results.append(result)
        
        # Sort by depth
        valid_results = [r for r in results if not np.isnan(r["depth"])]
        invalid_results = [r for r in results if np.isnan(r["depth"])]
        
        # Sort valid results by depth (nearest first)
        valid_results.sort(key=lambda x: x["depth"])
        
        # Combine results and add ranks
        final_results = valid_results + invalid_results
        for i, result in enumerate(final_results):
            result["distance_rank"] = i + 1
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in depth calculation: {str(e)}")
        return [] 