from google import genai
from google.genai import types

class GeminiAPI:
    """
    A class for interacting with Google's Gemini API to process images and return bounding box data.

    Attributes:
        api_key (str): The API key for authentication with Google Gemini API.
        model_name (str): The specific Gemini model being used.
        prompt (str): The text prompt guiding the model's response.
        img (any): The image input to be processed by the model.
        safety_settings (list): A set of safety rules to filter harmful content.
        bounding_box_system_instructions (str): Instructions for the API to return bounding boxes in JSON format.
    """

    def __init__(self, api_key: str, model_name: str, prompt: str, img: any):
        """
        Initializes the GeminiAPI object with necessary parameters.

        Args:
            api_key (str): The API key for accessing Google Gemini API.
            model_name (str): The name of the AI model to use.
            prompt (str): A textual prompt that guides the AI's response.
            img (any): The image input to be analyzed.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = prompt
        self.img = img
        
        # Define safety settings to block potentially harmful content.
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]

        # Instruction to force API response in structured JSON format with bounding boxes.
        self.bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """

    def call_api(self):
        """
        Sends a request to the Gemini API with the given image and prompt.

        The method:
        1. Creates an authenticated API client.
        2. Sends the image and prompt data to the model.
        3. Requests a structured JSON response containing labeled bounding boxes.

        Returns:
            API response containing detected objects and bounding boxes in JSON format.
        """
        try:
            # Initialize API client using provided API key
            client = genai.Client(api_key=self.api_key)

            # Send request to the model and retrieve response
            response = client.models.generate_content(
                model=self.model_name,
                contents=[self.prompt, self.img],
                config=types.GenerateContentConfig(
                    system_instruction=self.bounding_box_system_instructions,
                    temperature=0,  # Keep responses consistent (less randomness)
                    safety_settings=self.safety_settings,
                )
            )

            return response  # Return the structured JSON response

        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None  # Return None in case of failure
