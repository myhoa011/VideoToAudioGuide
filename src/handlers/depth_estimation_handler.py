from src.utils.logger import logger
from src.helpers.depth_helper import predict
from src.initializer import initializer

class DepthEstimationHandler:
    """Handler for depth estimation of detected objects"""
    
    def __init__(self):
        """Initialize depth estimation handler"""
        
        self.depth_model = initializer.get_depth_model()
        
    def estimate_depths(self, objects: list, image_path: str) -> list:
        """
        Estimate depths for detected objects
        
        Args:
            objects (list): List of detected objects with bounding boxes
            image_path (str): Path to the image file
            
        Returns:
            list: Objects with added depth information
        """
        try:
            
            # Use depth model to estimate depths
            results = predict(self.depth_model, objects, image_path)
            
            # Log results for debugging
            logger.debug(f"Depth estimation results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in depth estimation: {str(e)}")
            return [] 