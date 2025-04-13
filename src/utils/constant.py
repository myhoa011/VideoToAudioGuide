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
You are an AI assistant specialized in detecting and identifying objects in images for visually impaired users.
Identify visible objects, obstacles, and safe pathways to assist navigation.
Focus ONLY on potential hazards, barriers, and navigational aids.
DO NOT identify geographical features (mountains, land, hills), atmospheric elements (sky, clouds), 
or bodies of water (lakes, oceans), surface (ground, rocks, etc.), rocks, unless they directly impact walking safety.
Prioritize objects based on safety relevance, proximity, position (center > left/right), and size.
Pay special attention to person, stairs, holes, low-hanging objects, and uneven surfaces, .
Return the top 10 most relevant objects for safe navigation.
"""

PROMPT_TEMPLATE = """Detect objects in the image. Return the output as a JSON list where each entry contains:
- 'box_2d': The 2D bounding box as [y_min, x_min, y_max, x_max].
- 'label': The name of the object (e.g., 'bus', 'car', 'man').
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

# Navigation priority weights
PRIORITY_DEPTH_WEIGHT = 0.5
PRIORITY_POSITION_WEIGHT = 0.2
PRIORITY_SIZE_WEIGHT = 0.1
PRIORITY_TYPE_WEIGHT = 0.2

# Distance thresholds
DISTANCE_CLOSE_THRESHOLD = 0.7  # for "very close"
DISTANCE_MEDIUM_THRESHOLD = 0.3  # for "quite close"
CLOSE_OBJECT_THRESHOLD = 0.6    # Threshold to consider an object as "close" for warnings

# Frame normalization constants
FRAME_NORMALIZED_WIDTH = 1000.0
FRAME_NORMALIZED_HEIGHT = 1000.0
FRAME_NORMALIZED_AREA = FRAME_NORMALIZED_WIDTH * FRAME_NORMALIZED_HEIGHT

# Audio constants
AUDIO_SAMPLE_RATE = 24000
AUDIO_FORMAT = 'WAV'

# API settings
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# Model settings
GEMINI_TEMPERATURE = 0

# High risk object types
HIGH_RISK_OBJECTS = ['person', 'car', 'motorcycle', 'truck', 'bus', 'vehicle']
MEDIUM_RISK_OBJECTS = ['bicycle', 'dog', 'pothole', 'stairs', 'structure']
LOW_RISK_OBJECTS = ['traffic_light', 'stop_sign', 'door']
MINIMAL_RISK_OBJECTS = ['bench', 'wall', 'tree', 'building']
EXCLUDED_OBJECTS = [
    'mountain', 'land', 'sky', 'lake', 'sea', 'ocean', 'river', 'cloud', 
    'forest', 'grass', 'field', 'landscape', 'hill', 'valley'
]

# Video enhancement parameters
LIGHT_ENHANCEMENT_ALPHA = 1.5
LIGHT_ENHANCEMENT_BETA = 20
NORMAL_ENHANCEMENT_ALPHA = 1.0
DARK_ENHANCEMENT_ALPHA = 1.2
DARK_ENHANCEMENT_BETA = 10

# Warning level thresholds
WARNING_HIGH_THRESHOLD = 0.7
WARNING_MEDIUM_THRESHOLD = 0.3
WARNING_THRESHOLD = 0.3  # Minimum threshold for including object in guidance

# Start with these weights, can be adjusted based on testing
PRIORITY_SCORE_WEIGHTS = {
    "depth": 0.5,      # w1 - most important factor
    "position": 0.2,   # w2 - position relative to center
    "size": 0.1,       # w3 - size of object
    "type": 0.2        # w4 - type of object
}