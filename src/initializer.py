import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from transformers import pipeline
from google import genai
from openai import OpenAI
from aiogtts import aiogTTS
import kokoro

from src.utils.logger import reset_logger, logger
from src.utils.constant import (
    OUTPUT_PATH, OUTPUT_FRAME_PATH, OUTPUT_AUDIO_PATH, 
    DEPTH_MODEL, OUTPUT_REPORTS_PATH,
    KOKORO_REPO_ID
)
reset_logger()
load_dotenv()

class Initializer:
    """Initialize and manage models"""
    
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Initializer, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not Initializer._is_initialized:
            self.depth_model = None
            self.gemini_client = None
            self.openai_client = None
            self.kokoro_pipeline = None
            self.aiogTTS_engine = None
            Initializer._is_initialized = True

    async def initialize_models(self):
        """Initialize all models and resources"""
        try:
            logger.info("Starting model initialization...")
            
            # Initialize depth estimation model
            await self._init_depth_model()
            
            # Initialize Gemini client
            await self._init_gemini_client()
            
            # Initialize OpenAI client
            await self._init_openai_client()
            
            # Initialize TTS engines
            await self._init_tts_engines()
            
            # Create output directories
            self._create_output_dirs()
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
            
    async def _init_depth_model(self):
        """Initialize depth estimation model"""
        try:
            model_name = DEPTH_MODEL
            logger.info(f"Loading depth model: {model_name}")
            
            self.depth_model = pipeline(
                task="depth-estimation",
                model=model_name
            )
            
            logger.info("Depth model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading depth model: {str(e)}")
            raise
            
    async def _init_gemini_client(self):
        """Initialize Gemini API client"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not configured")
                
            logger.info("Initializing Gemini client")
            self.gemini_client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            raise
            
    async def _init_openai_client(self):
        """Initialize OpenAI API client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")
                
            logger.info("Initializing OpenAI client")
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    async def _init_tts_engines(self):
        """Initialize Text-to-Speech engines"""
        try:
            # Initialize Kokoro TTS
            try:
                logger.info("Initializing Kokoro TTS engine")
                self.kokoro_pipeline = kokoro.KPipeline(lang_code='a', repo_id=KOKORO_REPO_ID)  # 'a' for American English
                
                logger.info("Kokoro TTS engine initialized successfully")
            except Exception as e:
                logger.warning(f"Error initializing Kokoro TTS engine: {str(e)}")
                self.kokoro_pipeline = None
            
            # Initialize aiogTTS
            try:
                logger.info("Initializing aiogTTS engine")
                self.aiogTTS_engine = aiogTTS()
                logger.info("aiogTTS engine initialized successfully")
            except Exception as e:
                logger.warning(f"Error initializing aiogTTS engine: {str(e)}")
                self.aiogTTS_engine = None
                
        except Exception as e:
            logger.error(f"Error initializing TTS engines: {str(e)}")
            # Don't raise exception here - TTS engines are optional
            
    def _create_output_dirs(self):
        """Create necessary output directories"""
        try:
            dirs = [
                OUTPUT_PATH,
                OUTPUT_FRAME_PATH,
                OUTPUT_AUDIO_PATH,
                OUTPUT_REPORTS_PATH,
                "logs/",
            ]
            
            for dir_path in dirs:
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
                
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
            
    def get_depth_model(self):
        """Get instance of depth model"""
        if not self.depth_model:
            raise RuntimeError("Depth model not initialized")
        return self.depth_model
        
    def get_gemini_client(self):
        """Get instance of Gemini client"""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")
        return self.gemini_client
        
    def get_openai_client(self):
        """Get instance of OpenAI client"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        return self.openai_client
    
    def get_kokoro_pipeline(self):
        """Get instance of Kokoro pipeline"""
        return self.kokoro_pipeline
    
    def get_aiogTTS_engine(self):
        """Get instance of aiogTTS engine"""
        return self.aiogTTS_engine

initializer = Initializer()
asyncio.run(initializer.initialize_models())