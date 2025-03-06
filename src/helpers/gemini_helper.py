import re
from google.genai import types
from PIL import Image
from io import BytesIO
import aiofiles
import asyncio
import json
from src.utils.logger import logger

async def call_api(gemini_client, prompt, system_instructions, safety_settings, model_name, img_path: str) -> list:
    """
    Call Gemini API to detect objects in image
    
    Args:
        img_path (str): Path to image file
        
    Returns:
        list: Detected objects with bounding boxes
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
    
def extract_json_from_response(response_text: str) -> dict:
    """
    Extracts and parses JSON data from a Gemini API response.

    This function removes unnecessary markdown formatting and extracts valid JSON.

    Args:
        response_text (str): Raw response text from the Gemini API.

    Returns:
        dict: Parsed JSON data if extraction is successful, else an empty dictionary.
    """
    try:

        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```", 1)[1].split("```")[0].strip()

        response_text = re.sub(r"(\r\n|\r|\n)", "", response_text).strip()

        json_data = json.loads(response_text)
        return json_data

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from response: {str(e)}")
        return {}

    except Exception as e:
        logger.error(f"Unexpected error extracting JSON: {str(e)}")
        return {}
