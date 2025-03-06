from google import genai
from google.genai import types
from PIL import Image
import asyncio
import json
import os

from src.utils.logger import logger
from src.helpers.gemini_helper import call_api


class ObjectDetectionHandler:
    """Handler for object detection using Gemini API"""
    
    def __init__(self, gemini_client, model_name="gemini-2.0-flash"):
        """
        Initialize the handler
        """
        self.gemini_client = gemini_client
        self.prompt = os.getenv("PROMPT_TEMPLATE")
        self.model_name = model_name
        # Safety settings
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        # System instructions
        self.system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """
        
    def detect_objects(self, image_path):
        return call_api(self.gemini_client, self.prompt, self.system_instructions, self.safety_settings, self.model_name, image_path)

