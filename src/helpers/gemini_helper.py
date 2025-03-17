import re
from google.genai import types
from PIL import Image
from io import BytesIO
import aiofiles
import asyncio
import json
from typing import List
from src.utils.logger import logger
from schemas import DetectedObject
async def call_api(gemini_client, prompt, system_instructions, safety_settings, model_name, img_path: str) -> List[DetectedObject]:
    """
    Call Gemini API to detect objects in image
    
    Args:
        img_path (str): Path to image file
        
    Returns:
        List[DetectedObject]: Detected objects with bounding boxes
    """
    try:
        # Read image file
        async with aiofiles.open(img_path, mode='rb') as file:
            content = await file.read()
            img = Image.open(BytesIO(content))

        # Call Gemini API
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model_name,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=0,
                safety_settings=safety_settings,
            )
        )
        return extract_json_from_response(response.text)

    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return []
        logger.error(f"Error calling Gemini API: {str(e)}")
        return [] 
    
def extract_json_from_response(response_text: str) -> List[DetectedObject]:
    """
    Extracts and parses JSON data from a Gemini API response, converting to DetectedObject instances.

    This function removes unnecessary markdown formatting and extracts valid JSON.

    Args:
        response_text (str): Raw response text from the Gemini API.

    Returns:
        List[DetectedObject]: List of DetectedObject instances if extraction is successful, else an empty list.
    """
    try:
        # Extract JSON from markdown if needed
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```", 1)[1].split("```")[0].strip()

        response_text = re.sub(r"(\r\n|\r|\n)", "", response_text).strip()

        # Parse JSON array
        json_data = json.loads(response_text)
        
        # Convert each item to DetectedObject
        detected_objects = []
        for item in json_data:
            try:
                detected_objects.append(DetectedObject(
                    box_2d=item.get('box_2d', [0, 0, 0, 0]),
                    label=item.get('label', ''),
                    position=item.get('position', 'unknown'),
                    type=item.get('type', 'unknown')
                ))
            except Exception as e:
                logger.error(f"Error parsing detected object: {str(e)}")
                
        return detected_objects

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from response: {str(e)}")
        return []

    except Exception as e:
        logger.error(f"Unexpected error extracting JSON: {str(e)}")
        return []
