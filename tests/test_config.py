#!/usr/bin/env python3
"""Test configuration system functionality."""

import os
import tempfile
from pathlib import Path

# Test basic configuration loading
def test_config_basic():
    """Test basic configuration loading without files."""
    from docparse.config import DocparseConfig, reset_config
    
    reset_config()
    config = DocparseConfig()
    
    # Test defaults
    assert config.layout_detection.dpi == 224
    assert config.text_extraction.line_clustering_tolerance == 0.25
    assert config.runtime.execution_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("✓ Basic configuration defaults work")


def test_config_toml():
    """Test TOML configuration loading."""
    from docparse.config import DocparseConfig, reset_config
    
    # Create a temporary config file
    config_content = """
[text_extraction]
line_clustering_tolerance = 0.3
space_threshold_height = 0.15

[layout_detection] 
dpi = 300
confidence_threshold = 0.4
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        reset_config()
        config = DocparseConfig.load_from_toml(temp_path)
        
        # Test loaded values
        assert config.text_extraction.line_clustering_tolerance == 0.3
        assert config.text_extraction.space_threshold_height == 0.15
        assert config.layout_detection.dpi == 300
        assert config.layout_detection.confidence_threshold == 0.4
        
        print("✓ TOML configuration loading works")
        
    finally:
        os.unlink(temp_path)


def test_env_overrides():
    """Test environment variable overrides."""
    from docparse.config import DocparseConfig, reset_config
    
    # Set environment variables
    os.environ['DOCPARSE_LAYOUT_DETECTION__DPI'] = '150'
    os.environ['DOCPARSE_TEXT_EXTRACTION__LINE_CLUSTERING_TOLERANCE'] = '0.35'
    
    try:
        reset_config()
        config = DocparseConfig()
        
        assert config.layout_detection.dpi == 150
        assert config.text_extraction.line_clustering_tolerance == 0.35
        
        print("✓ Environment variable overrides work")
        
    finally:
        # Clean up environment
        os.environ.pop('DOCPARSE_LAYOUT_DETECTION__DPI', None)
        os.environ.pop('DOCPARSE_TEXT_EXTRACTION__LINE_CLUSTERING_TOLERANCE', None)


def test_config_validation():
    """Test configuration validation."""
    from docparse.config_validator import validate_config, ConfigValidationError
    from docparse.config import DocparseConfig
    
    # Test valid config
    config = DocparseConfig()
    warnings = validate_config(config)
    print(f"✓ Valid config has {len(warnings)} warnings")
    
    # Test invalid DPI
    config.layout_detection.dpi = 1000  # Too high
    warnings = validate_config(config)
    assert any("DPI value 1000" in w for w in warnings)
    print("✓ DPI validation works")
    
    # Test invalid confidence threshold
    config.layout_detection.confidence_threshold = 1.5  # Too high
    warnings = validate_config(config)
    assert any("Confidence threshold 1.5" in w for w in warnings)
    print("✓ Confidence threshold validation works")


def test_profiles():
    """Test configuration profiles."""
    from docparse.config import DocparseConfig, reset_config
    
    # Create config with profiles
    config_content = """
[layout_detection]
dpi = 224
confidence_threshold = 0.3

[profiles.academic.layout_detection]
dpi = 300
confidence_threshold = 0.4

[profiles.fast.layout_detection]
dpi = 150
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        reset_config()
        config = DocparseConfig.load_from_toml(temp_path)
        
        # Test base config
        assert config.layout_detection.dpi == 224
        assert config.layout_detection.confidence_threshold == 0.3
        
        # Test academic profile
        academic_config = config.apply_profile("academic")
        assert academic_config.layout_detection.dpi == 300
        assert academic_config.layout_detection.confidence_threshold == 0.4
        
        # Test fast profile  
        fast_config = config.apply_profile("fast")
        assert fast_config.layout_detection.dpi == 150
        
        print("✓ Configuration profiles work")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("Testing configuration system...")
    
    test_config_basic()
    test_config_toml()
    test_env_overrides()
    test_config_validation()
    test_profiles()
    
    print("\n✅ All configuration tests passed!")