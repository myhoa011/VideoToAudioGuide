
from google.genai import types

from src.utils.logger import logger
from src.helpers.gemini_helper import call_api
from src.utils.constant import PROMPT_TEMPLATE, CATEGORY, THRESHOLD, SYSTEM_INSTRUCTON, GEMINI_MODEL_NAME
from src.initializer import initializer

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
        self.prompt = PROMPT_TEMPLATE
        self.model_name = model_name
        # Safety settings
        self.safety_settings = [
            types.SafetySetting(
                category=CATEGORY,
                threshold=THRESHOLD,
            ),
        ]
        # System instructions
        self.system_instructions = SYSTEM_INSTRUCTON
        
    async def detect_objects(self, image_path: str) -> list:
        """
        Detect objects in an image using Gemini API
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            list: Detected objects with bounding boxes
        """

        return await call_api(
            self.gemini_client,
            self.prompt,
            self.system_instructions,
            self.safety_settings,
            self.model_name,
            image_path
        )

