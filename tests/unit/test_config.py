"""Unit tests for configuration system."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from docparse.config import (
    DocparseConfig,
    TextExtractionConfig,
    LayoutDetectionConfig,
    RenderingConfig,
    RuntimeConfig,
    PathsConfig,
    reset_config,
    init_config,
    get_config,
    get_dpi,
    get_model_path,
    get_confidence_threshold
)
from docparse.config_validator import (
    validate_config,
    safe_init_config,
    ConfigValidationError
)


@pytest.mark.unit
class TestConfigModels:
    """Test configuration data models."""
    
    def test_text_extraction_config_defaults(self):
        """Test default values for text extraction config."""
        config = TextExtractionConfig()
        assert config.line_clustering_tolerance == 0.25
        assert config.space_threshold_height == 0.20
        assert config.space_threshold_width == 0.40
        assert config.min_character_size == 0.0
    
    def test_text_extraction_config_validation(self):
        """Test validation of text extraction config."""
        # Valid config
        config = TextExtractionConfig(
            line_clustering_tolerance=0.5,
            space_threshold_height=0.3,
            space_threshold_width=0.6,
            min_character_size=1.0
        )
        assert config.line_clustering_tolerance == 0.5
        
        # Invalid values should raise validation errors
        with pytest.raises(ValueError):
            TextExtractionConfig(line_clustering_tolerance=-0.1)  # Below 0
        
        with pytest.raises(ValueError):
            TextExtractionConfig(line_clustering_tolerance=1.1)   # Above 1
    
    def test_layout_detection_config_defaults(self):
        """Test default values for layout detection config."""
        config = LayoutDetectionConfig()
        assert config.dpi == 224
        assert config.model_path == "yolov8s-doclaynet.onnx"
        assert config.input_size == 1024
        assert config.confidence_threshold == 0.3
        assert config.iou_threshold == 0.45
        assert config.padding_color == [114, 114, 114]
    
    def test_layout_detection_config_validation(self):
        """Test validation of layout detection config."""
        # Valid config
        config = LayoutDetectionConfig(dpi=300, confidence_threshold=0.5)
        assert config.dpi == 300
        assert config.confidence_threshold == 0.5
        
        # Invalid DPI
        with pytest.raises(ValueError):
            LayoutDetectionConfig(dpi=50)  # Below minimum
        
        with pytest.raises(ValueError):
            LayoutDetectionConfig(dpi=700)  # Above maximum
        
        # Invalid confidence threshold
        with pytest.raises(ValueError):
            LayoutDetectionConfig(confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            LayoutDetectionConfig(confidence_threshold=1.1)
    
    def test_runtime_config_defaults(self):
        """Test default values for runtime config."""
        config = RuntimeConfig()
        assert config.execution_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert config.thread_count == 0
        assert config.memory_limit_mb == 0


@pytest.mark.unit
class TestDocparseConfig:
    """Test main configuration class."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = DocparseConfig()
        
        assert isinstance(config.text_extraction, TextExtractionConfig)
        assert isinstance(config.layout_detection, LayoutDetectionConfig)
        assert isinstance(config.rendering, RenderingConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.paths, PathsConfig)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "text_extraction": {
                "line_clustering_tolerance": 0.3,
                "space_threshold_height": 0.15
            },
            "layout_detection": {
                "dpi": 300,
                "confidence_threshold": 0.4
            }
        }
        
        config = DocparseConfig.from_dict(config_dict)
        assert config.text_extraction.line_clustering_tolerance == 0.3
        assert config.text_extraction.space_threshold_height == 0.15
        assert config.layout_detection.dpi == 300
        assert config.layout_detection.confidence_threshold == 0.4
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = DocparseConfig()
        config_dict = config.to_dict()
        
        assert "text_extraction" in config_dict
        assert "layout_detection" in config_dict
        assert "rendering" in config_dict
        assert "runtime" in config_dict
        assert "paths" in config_dict
        
        # Check nested structure
        assert "line_clustering_tolerance" in config_dict["text_extraction"]
        assert "dpi" in config_dict["layout_detection"]
    
    def test_load_from_toml(self, temp_config_file):
        """Test loading configuration from TOML file."""
        config = DocparseConfig.load_from_toml(temp_config_file)
        
        assert config.text_extraction.line_clustering_tolerance == 0.3
        assert config.text_extraction.space_threshold_height == 0.15
        assert config.layout_detection.dpi == 300
        assert config.layout_detection.confidence_threshold == 0.4
        assert config.runtime.execution_providers == ["CPUExecutionProvider"]
    
    def test_load_from_nonexistent_toml(self):
        """Test loading from non-existent TOML file."""
        with pytest.raises(FileNotFoundError):
            DocparseConfig.load_from_toml("nonexistent.toml")
    
    def test_load_from_invalid_toml(self, temp_dir):
        """Test loading from invalid TOML file."""
        invalid_toml = temp_dir / "invalid.toml"
        invalid_toml.write_text("invalid toml content [[[")
        
        with pytest.raises(Exception):  # TOML parsing error
            DocparseConfig.load_from_toml(invalid_toml)
    
    def test_apply_profile(self, temp_config_file):
        """Test applying configuration profiles."""
        config = DocparseConfig.load_from_toml(temp_config_file)
        
        # Test academic profile
        academic_config = config.apply_profile("academic")
        assert academic_config.layout_detection.confidence_threshold == 0.5
        assert academic_config.layout_detection.dpi == 300
        assert academic_config.text_extraction.line_clustering_tolerance == 0.20
        
        # Test fast profile
        fast_config = config.apply_profile("fast")
        assert fast_config.layout_detection.dpi == 150
        assert fast_config.layout_detection.input_size == 512
        assert fast_config.runtime.execution_providers == ["CPUExecutionProvider"]
    
    def test_apply_nonexistent_profile(self):
        """Test applying non-existent profile."""
        config = DocparseConfig()
        
        # Should return the same config if profile doesn't exist
        result_config = config.apply_profile("nonexistent")
        assert result_config == config
    
    def test_get_model_path(self):
        """Test model path resolution."""
        config = DocparseConfig()
        model_path = config.get_model_path()
        
        assert isinstance(model_path, Path)
        assert model_path.name == "yolov8s-doclaynet.onnx"
        assert str(model_path).startswith("models/")


@pytest.mark.unit
class TestEnvironmentVariables:
    """Test environment variable integration."""
    
    def test_env_var_override(self, mock_env_vars):
        """Test environment variable overrides."""
        config = DocparseConfig()
        
        assert config.layout_detection.dpi == 200
        assert config.text_extraction.line_clustering_tolerance == 0.35
        assert config.runtime.execution_providers == ["CPUExecutionProvider"]
    
    def test_env_var_precedence(self, temp_config_file, mock_env_vars):
        """Test that environment variables override TOML config."""
        # TOML file has dpi=300, but env var has dpi=200
        config = DocparseConfig.load_from_toml(temp_config_file)
        
        # Environment variables should take precedence
        assert config.layout_detection.dpi == 200
        assert config.text_extraction.line_clustering_tolerance == 0.35
    
    def test_invalid_env_var_type(self):
        """Test handling of invalid environment variable types."""
        with patch.dict(os.environ, {'DOCPARSE_LAYOUT_DETECTION__DPI': 'not_a_number'}):
            with pytest.raises(ValueError):
                DocparseConfig()
    
    def test_invalid_env_var_json(self):
        """Test handling of invalid JSON in environment variables."""
        with patch.dict(os.environ, {'DOCPARSE_RUNTIME__EXECUTION_PROVIDERS': 'invalid_json'}):
            with pytest.raises(ValueError):
                DocparseConfig()


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config = DocparseConfig()
        warnings = validate_config(config)
        
        # Should have warnings about missing model file but no critical errors
        assert isinstance(warnings, list)
    
    def test_missing_model_validation(self):
        """Test validation with missing model file."""
        config = DocparseConfig()
        warnings = validate_config(config)
        
        # Should warn about missing default model
        model_warnings = [w for w in warnings if "model not found" in w.lower()]
        assert len(model_warnings) > 0
    
    def test_invalid_dpi_validation(self):
        """Test validation of invalid DPI values."""
        config = DocparseConfig()
        config.layout_detection.dpi = 1000  # Too high
        
        warnings = validate_config(config)
        dpi_warnings = [w for w in warnings if "dpi" in w.lower()]
        assert len(dpi_warnings) > 0
    
    def test_invalid_confidence_validation(self):
        """Test validation of invalid confidence threshold."""
        config = DocparseConfig()
        config.layout_detection.confidence_threshold = 1.5  # Too high
        
        warnings = validate_config(config)
        conf_warnings = [w for w in warnings if "confidence" in w.lower()]
        assert len(conf_warnings) > 0
    
    def test_safe_init_config_success(self, temp_config_file):
        """Test successful safe configuration initialization."""
        safe_init_config(config_path=temp_config_file, profile="academic", strict=False)
        
        config = get_config()
        assert config.layout_detection.confidence_threshold == 0.5
    
    def test_safe_init_config_strict_failure(self):
        """Test safe configuration initialization with strict validation."""
        with pytest.raises(ConfigValidationError):
            safe_init_config(config_path=None, profile=None, strict=True)
    
    def test_safe_init_config_invalid_profile(self):
        """Test safe configuration initialization with invalid profile."""
        with pytest.raises(ConfigValidationError):
            safe_init_config(config_path=None, profile="nonexistent", strict=True)


@pytest.mark.unit
class TestConfigGlobals:
    """Test global configuration functions."""
    
    def test_init_config_default(self):
        """Test default configuration initialization."""
        init_config()
        config = get_config()
        
        assert isinstance(config, DocparseConfig)
        assert config.layout_detection.dpi == 224
    
    def test_init_config_with_file(self, temp_config_file):
        """Test configuration initialization with file."""
        init_config(config_path=temp_config_file)
        config = get_config()
        
        assert config.layout_detection.dpi == 300
        assert config.text_extraction.line_clustering_tolerance == 0.3
    
    def test_init_config_with_profile(self, temp_config_file):
        """Test configuration initialization with profile."""
        init_config(config_path=temp_config_file, profile="fast")
        config = get_config()
        
        assert config.layout_detection.dpi == 150
        assert config.layout_detection.input_size == 512
    
    def test_get_dpi(self):
        """Test DPI getter function."""
        init_config()
        dpi = get_dpi()
        assert isinstance(dpi, int)
        assert dpi == 224
    
    def test_get_model_path(self):
        """Test model path getter function."""
        init_config()
        model_path = get_model_path()
        assert isinstance(model_path, Path)
    
    def test_get_confidence_threshold(self):
        """Test confidence threshold getter function."""
        init_config()
        threshold = get_confidence_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
    
    def test_reset_config(self, temp_config_file):
        """Test configuration reset."""
        # Initialize with custom config
        init_config(config_path=temp_config_file)
        config = get_config()
        assert config.layout_detection.dpi == 300
        
        # Reset and check defaults are restored
        reset_config()
        config = get_config()
        assert config.layout_detection.dpi == 224


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases and error handling."""
    
    def test_config_with_empty_toml(self, temp_dir):
        """Test loading empty TOML file."""
        empty_toml = temp_dir / "empty.toml"
        empty_toml.write_text("")
        
        config = DocparseConfig.load_from_toml(empty_toml)
        # Should use defaults
        assert config.layout_detection.dpi == 224
    
    def test_config_with_partial_toml(self, temp_dir):
        """Test loading partial TOML file."""
        partial_toml = temp_dir / "partial.toml"
        partial_toml.write_text("""
[layout_detection]
dpi = 300
""")
        
        config = DocparseConfig.load_from_toml(partial_toml)
        assert config.layout_detection.dpi == 300
        # Other values should be defaults
        assert config.text_extraction.line_clustering_tolerance == 0.25
    
    def test_config_serialization_roundtrip(self):
        """Test configuration serialization and deserialization."""
        original_config = DocparseConfig()
        original_config.layout_detection.dpi = 300
        original_config.text_extraction.line_clustering_tolerance = 0.3
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        
        # Deserialize from dict
        restored_config = DocparseConfig.from_dict(config_dict)
        
        assert restored_config.layout_detection.dpi == 300
        assert restored_config.text_extraction.line_clustering_tolerance == 0.3
    
    def test_config_copy(self):
        """Test configuration copying."""
        original_config = DocparseConfig()
        original_config.layout_detection.dpi = 300
        
        copied_config = original_config.copy()
        
        # Modify original
        original_config.layout_detection.dpi = 400
        
        # Copy should be unchanged
        assert copied_config.layout_detection.dpi == 300
    
    def test_thread_safety(self):
        """Test basic thread safety of configuration access."""
        import threading
        import time
        
        results = []
        
        def worker():
            init_config()
            config = get_config()
            results.append(config.layout_detection.dpi)
            time.sleep(0.1)
            results.append(config.layout_detection.dpi)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All results should be the same (224 - default DPI)
        assert all(dpi == 224 for dpi in results)
        assert len(results) == 10  # 2 results per thread, 5 threads