"""
Configuration settings for the Prior Authorization AI Tool.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DATA_DIR = BASE_DIR / "Input Data"
OUTPUT_DIR = BASE_DIR / "Output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class AIConfig:
    """Configuration for AI models."""
    # Supported providers: "openai", "anthropic"
    provider: str = "openai"
    
    # OpenAI settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = "gpt-4o"  # Vision-capable model
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_model: str = "claude-sonnet-4-20250514"
    
    # Processing settings
    max_tokens: int = 4096
    temperature: float = 0.1  # Low temperature for more deterministic extraction


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing."""
    # DPI for PDF to image conversion (higher = better quality but slower)
    pdf_dpi: int = 200
    
    # Whether to use OCR for scanned documents
    use_ocr: bool = True
    
    # Chunk size for RAG-style retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class AppConfig:
    """Main application configuration."""
    ai: AIConfig = field(default_factory=AIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    input_dir: Path = INPUT_DATA_DIR
    output_dir: Path = OUTPUT_DIR
    
    # Logging
    log_level: str = "INFO"
    
    # Backend API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Default configuration instance
config = AppConfig()
