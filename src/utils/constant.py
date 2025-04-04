# Output paths
OUTPUT_PATH = "outputs/"
OUTPUT_FRAME_PATH = OUTPUT_PATH + "frames/"
OUTPUT_AUDIO_PATH = OUTPUT_PATH + "audio/"
OUTPUT_REPORTS_PATH = OUTPUT_PATH + "reports/"

# Video processing
TIME_INTERVAL = 1

# Depth estimation model
DEPTH_MODEL= "depth-anything/Depth-Anything-V2-Small-hf"

# Text-to-Speech Constants
TTS_ENGINE_OPENAI = "openai"
TTS_ENGINE_GTTS = "gtts"
TTS_ENGINE_KOKORO = "kokoro"

TTS_ENGINES = [TTS_ENGINE_OPENAI, TTS_ENGINE_GTTS, TTS_ENGINE_KOKORO]

# OpenAI TTS
OPENAI_MODEL_NAME = "tts-1"
OPENAI_TTS_VOICE = "alloy"

# gTTS settings
GTTS_LANGUAGE = "en"

# Kokoro settings
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"

# Gemini API
GEMINI_MODEL_NAME = "gemini-2.0-flash"
CATEGORY = "HARM_CATEGORY_DANGEROUS_CONTENT"
THRESHOLD = "BLOCK_ONLY_HIGH"

# Prompts
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in detecting and identifying objects in images.
Your task is to identify all visible objects in the image and provide their bounding boxes.
Please return a maximum of 10 detected objects.
"""

PROMPT_TEMPLATE = """Detect objects in the image. Return the output as a JSON list where each entry contains:
- 'box_2d': The 2D bounding box as [y_min, x_min, y_max, x_max].
- 'label': The general name of the object (e.g., 'bus', 'car', 'person').
- 'position': The relative position of the object in the image ('left', 'right', 'center').
- 'type': The category or type of the object (e.g., 'vehicle', 'animal', 'person', 'building', etc.).

Ensure that:
- 'label' only contains the object's general category, not specific attributes.
- 'position' describes the object's placement.
- 'type' describes the broader classification of the object.

Do not return Markdown, explanations, or extra text. Return ONLY the JSON."""

#Process video

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

CONCURRENCY_LIMIT = 5