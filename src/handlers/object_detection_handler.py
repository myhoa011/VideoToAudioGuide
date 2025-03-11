"""
Handler for object detection functionality.
"""

from google import genai
from google.genai import types
from PIL import Image
import asyncio
import json
import os

from src.utils.logger import logger
from src.helpers.gemini_helper import call_api
from src.utils.constant import PROMPT_TEMPLATE, CATEGORY, THRESHOLD, SYSTEM_INSTRUCTON, MODEL_NAME
from src.initializer import get_initializer

class ObjectDetectionHandler:
    """Handler for object detection using Gemini API"""
    
    def __init__(self, gemini_client=None, model_name=MODEL_NAME):
        """
        Initialize the handler
        
        Args:
            gemini_client: Optional pre-initialized Gemini client
            model_name: Name of the model to use
        """
        self.gemini_client = gemini_client
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
        # Use provided client or get from initializer
        if self.gemini_client is None:
            initializer = get_initializer()
            self.gemini_client = initializer.get_gemini_client()
            
        return await call_api(
            self.gemini_client,
            self.prompt,
            self.system_instructions,
            self.safety_settings,
            self.model_name,
            image_path
        )

