"""Configuration settings loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the multi-agent system."""

    # API Configuration
    API_KEY = os.getenv('API_KEY', os.getenv('OPENROUTER_API_KEY'))
    API_BASE = os.getenv('API_BASE', 'https://openrouter.ai/api/v1')

    # Default Models
    DEFAULT_HOMOGENEOUS_MODEL = os.getenv('DEFAULT_HOMOGENEOUS_MODEL', 'meta-llama/llama-3.3-70b-instruct:free')
    DEFAULT_INDIVIDUAL_MODEL = os.getenv('DEFAULT_INDIVIDUAL_MODEL', 'qwen/qwen-2.5-72b-instruct:free')

    # Heterogeneous Models
    HETEROGENEOUS_MODERATOR_MODEL = os.getenv('HETEROGENEOUS_MODERATOR_MODEL', 'nousresearch/hermes-3-llama-3.1-405b:free')
    HETEROGENEOUS_DLB_MODEL = os.getenv('HETEROGENEOUS_DLB_MODEL', 'meta-llama/llama-3.3-70b-instruct:free')
    HETEROGENEOUS_PNM_MODEL = os.getenv('HETEROGENEOUS_PNM_MODEL', 'qwen/qwen-2.5-72b-instruct:free')

    # RAG Configuration
    RAG_DATA_PATH = os.getenv('RAG_DATA_PATH', './data/rag_dataset/')
    RAG_MODEL_PATH = os.getenv('RAG_MODEL_PATH', './models/embeddings_model/')

    # FAISS Configuration
    DEFAULT_K = int(os.getenv('DEFAULT_K', '2'))
    EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')

    # Rate Limiting
    SLEEP_BASE = float(os.getenv('SLEEP_BASE', '12'))
    SLEEP_JITTER = float(os.getenv('SLEEP_JITTER', '4'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '2'))
    RETRY_BASE_DELAY = float(os.getenv('RETRY_BASE_DELAY', '15'))

    # Debate Configuration
    MAX_ROUNDS = int(os.getenv('MAX_ROUNDS', '6'))
    MAX_MESSAGES = int(os.getenv('MAX_MESSAGES', '40'))
    MAX_TOKENS_PER_RESPONSE = int(os.getenv('MAX_TOKENS_PER_RESPONSE', '150'))

    # Output Configuration
    DEFAULT_OUTPUT_DIR = Path(os.getenv('DEFAULT_OUTPUT_DIR', './outputs'))
    DEFAULT_DECISION_DIR = Path(os.getenv('DEFAULT_DECISION_DIR', './decisions'))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required. Set it in the .env file or as environment variable.")

        if not Path(cls.RAG_DATA_PATH).exists():
            print(f"Warning: RAG data path does not exist: {cls.RAG_DATA_PATH}")

        if not Path(cls.RAG_MODEL_PATH).exists():
            print(f"Warning: RAG model path does not exist: {cls.RAG_MODEL_PATH}")