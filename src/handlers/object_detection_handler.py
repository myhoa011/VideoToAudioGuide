from google.genai import types
from typing import List

from src.utils.logger import logger
from src.helpers.gemini_helper import call_api
from src.utils.constant import PROMPT_TEMPLATE, CATEGORY, THRESHOLD, SYSTEM_INSTRUCTION, GEMINI_MODEL_NAME, EXCLUDED_OBJECTS
from src.initializer import initializer
from src.schemas.detection import DetectedObject


class ObjectDetectionHandler:
    """Handler for object detection using Gemini API"""
    
    def __init__(self, model_name=GEMINI_MODEL_NAME):
        """
        Initialize the handler
        
        Args:
            gemini_client: Gemini client
            model_name: Name of the model to use
        """
        self.gemini_client = initializer.get_gemini_client()
        self.model_name = model_name
        # Safety settings
        self.safety_settings = [
            types.SafetySetting(
                category=CATEGORY,
                threshold=THRESHOLD,
            ),
        ]
        
    async def detect_objects(self, image_path: str) -> List[DetectedObject]:
        """
        Detect objects in an image using Gemini API
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            list: Detected objects with bounding boxes
        """

        objects = await call_api(
            self.gemini_client,
            PROMPT_TEMPLATE,
            SYSTEM_INSTRUCTION,
            self.safety_settings,
            self.model_name,
            image_path
        )

        filtered_objects = [obj for obj in objects if obj.label not in EXCLUDED_OBJECTS and obj.type not in ['geographical feature', 'atmospheric', 'body of water', 'surface']]
        return filtered_objects

