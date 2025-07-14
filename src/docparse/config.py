"""Configuration management for docparse."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class TextExtractionConfig(BaseModel):
    """Configuration for text extraction parameters."""
    line_clustering_tolerance: float = Field(default=0.25, ge=0.0, le=1.0)
    space_threshold_height: float = Field(default=0.20, ge=0.0, le=1.0)
    space_threshold_width: float = Field(default=0.40, ge=0.0, le=1.0)
    min_character_size: float = Field(default=0.0, ge=0.0)


class LayoutDetectionConfig(BaseModel):
    """Configuration for layout detection parameters."""
    dpi: int = Field(default=224, ge=72, le=600)
    model_path: str = Field(default="yolov8s-doclaynet.onnx")
    input_size: int = Field(default=1024, ge=256)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    padding_color: List[int] = Field(default=[114, 114, 114])

    @field_validator('padding_color')
    @classmethod
    def validate_padding_color(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("padding_color must have exactly 3 RGB values")
        if not all(0 <= x <= 255 for x in v):
            raise ValueError("padding_color values must be between 0 and 255")
        return v


class RenderingConfig(BaseModel):
    """Configuration for PDF rendering parameters."""
    default_dpi: int = Field(default=224, ge=72, le=600)
    points_per_inch: float = Field(default=72.0, gt=0.0)
    auto_rotate_landscape: bool = Field(default=False)


class RuntimeConfig(BaseModel):
    """Configuration for runtime and execution parameters."""
    execution_providers: List[str] = Field(
        default=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    thread_count: int = Field(default=0, ge=0)
    memory_limit_mb: int = Field(default=0, ge=0)


class PathsConfig(BaseModel):
    """Configuration for file and directory paths."""
    pdfium_library_path: str = Field(default="./")
    models_directory: str = Field(default="models/")
    cache_directory: str = Field(default="")


class ProfileConfig(BaseModel):
    """Configuration profile with selective overrides."""
    text_extraction: Optional[TextExtractionConfig] = None
    layout_detection: Optional[LayoutDetectionConfig] = None
    rendering: Optional[RenderingConfig] = None
    runtime: Optional[RuntimeConfig] = None


class DocparseConfig(BaseSettings):
    """Main configuration class with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix='DOCPARSE_',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore'
    )

    text_extraction: TextExtractionConfig = Field(default_factory=TextExtractionConfig)
    layout_detection: LayoutDetectionConfig = Field(default_factory=LayoutDetectionConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict)

    @classmethod
    def load_from_toml(cls, config_path: Optional[Union[str, Path]] = None) -> "DocparseConfig":
        """Load configuration from TOML file."""
        if config_path is None:
            # Search for config file in order of preference
            config_paths = [
                Path("config.toml"),
                Path("./config.toml"),
                Path("docparse.toml"),
                Path("./docparse.toml"),
            ]
            
            for path in config_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                # No config file found, return default
                return cls()
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'rb') as f:
                config_data = tomllib.load(f)
            return cls(**config_data)
        except Exception as e:
            raise ValueError(f"Failed to parse configuration file {config_path}: {e}")

    def apply_profile(self, profile_name: str) -> "DocparseConfig":
        """Apply a configuration profile, returning a new instance."""
        if profile_name not in self.profiles:
            available = list(self.profiles.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {available}")
        
        profile = self.profiles[profile_name]
        
        # Create a copy of current config
        new_config = self.model_copy(deep=True)
        
        # Apply profile overrides
        if profile.text_extraction:
            new_config.text_extraction = profile.text_extraction
        if profile.layout_detection:
            new_config.layout_detection = profile.layout_detection
        if profile.rendering:
            new_config.rendering = profile.rendering
        if profile.runtime:
            new_config.runtime = profile.runtime
            
        return new_config

    def get_model_path(self) -> Path:
        """Get the full path to the layout detection model."""
        model_path = Path(self.layout_detection.model_path)
        if not model_path.is_absolute():
            model_path = Path(self.paths.models_directory) / model_path
        return model_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


# Global configuration instance
_config_instance: Optional[DocparseConfig] = None


def get_config() -> DocparseConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DocparseConfig.load_from_toml()
    return _config_instance


def init_config(
    config_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    **overrides: Any
) -> DocparseConfig:
    """Initialize the global configuration."""
    global _config_instance
    
    # Load from TOML file
    _config_instance = DocparseConfig.load_from_toml(config_path)
    
    # Apply profile if specified
    if profile:
        _config_instance = _config_instance.apply_profile(profile)
    
    # Apply any additional overrides
    if overrides:
        config_dict = _config_instance.to_dict()
        config_dict.update(overrides)
        _config_instance = DocparseConfig(**config_dict)
    
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None


# Helper functions for common configuration tasks
def get_dpi(cli_override: Optional[int] = None) -> int:
    """Get DPI setting with optional CLI override."""
    if cli_override is not None:
        return cli_override
    return get_config().layout_detection.dpi


def get_model_path(cli_override: Optional[Union[str, Path]] = None) -> Path:
    """Get model path with optional CLI override."""
    if cli_override is not None:
        return Path(cli_override)
    return get_config().get_model_path()


def get_confidence_threshold(cli_override: Optional[float] = None) -> float:
    """Get confidence threshold with optional CLI override."""
    if cli_override is not None:
        return cli_override
    return get_config().layout_detection.confidence_threshold


def get_execution_providers(cli_override: Optional[List[str]] = None) -> List[str]:
    """Get ONNX execution providers with optional CLI override."""
    if cli_override is not None:
        return cli_override
    return get_config().runtime.execution_providers