"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator
import pytest
import json
import shutil

from docparse.config import DocparseConfig, reset_config


@pytest.fixture(autouse=True)
def reset_config_before_test():
    """Reset configuration before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF file for testing."""
    return Path("sample.pdf")


@pytest.fixture
def config_toml_content() -> str:
    """Sample TOML configuration content."""
    return """
[text_extraction]
line_clustering_tolerance = 0.3
space_threshold_height = 0.15
space_threshold_width = 0.35
min_character_size = 1.0

[layout_detection]
dpi = 300
model_path = "models/yolov8s-doclaynet.onnx"
input_size = 1024
confidence_threshold = 0.4
iou_threshold = 0.5

[rendering]
default_dpi = 300
points_per_inch = 72.0
auto_rotate_landscape = true

[runtime]
execution_providers = ["CPUExecutionProvider"]
thread_count = 4
memory_limit_mb = 1024

[paths]
pdfium_library_path = "./"
models_directory = "models/"
cache_directory = "/tmp/docparse"

[profiles.academic.layout_detection]
confidence_threshold = 0.5
dpi = 300

[profiles.academic.text_extraction]
line_clustering_tolerance = 0.20

[profiles.fast.layout_detection]
dpi = 150
input_size = 512

[profiles.fast.runtime]
execution_providers = ["CPUExecutionProvider"]
"""


@pytest.fixture
def temp_config_file(temp_dir: Path, config_toml_content: str) -> Path:
    """Create a temporary configuration file."""
    config_file = temp_dir / "test_config.toml"
    config_file.write_text(config_toml_content)
    return config_file


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_env = {
        'DOCPARSE_LAYOUT_DETECTION__DPI': '200',
        'DOCPARSE_TEXT_EXTRACTION__LINE_CLUSTERING_TOLERANCE': '0.35',
        'DOCPARSE_RUNTIME__EXECUTION_PROVIDERS': '["CPUExecutionProvider"]'
    }
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_extracted_text() -> list:
    """Sample extracted text data for testing."""
    return [
        {
            "page": 0,
            "text": "Sample Document Title",
            "x0": 100.0,
            "y0": 700.0,
            "x1": 500.0,
            "y1": 720.0
        },
        {
            "page": 0,
            "text": "This is the first paragraph of the document.",
            "x0": 100.0,
            "y0": 650.0,
            "x1": 500.0,
            "y1": 670.0
        },
        {
            "page": 1,
            "text": "This is content on the second page.",
            "x0": 100.0,
            "y0": 650.0,
            "x1": 500.0,
            "y1": 670.0
        }
    ]


@pytest.fixture
def sample_layout_regions() -> list:
    """Sample layout detection regions for testing."""
    return [
        {
            "label": "Title",
            "confidence": 0.95,
            "bbox": [100, 700, 500, 720]
        },
        {
            "label": "Text",
            "confidence": 0.85,
            "bbox": [100, 650, 500, 670]
        },
        {
            "label": "Figure",
            "confidence": 0.90,
            "bbox": [150, 400, 450, 600]
        }
    ]


@pytest.fixture
def sample_json_lines(sample_extracted_text: list) -> str:
    """Sample JSON lines output for testing."""
    return '\n'.join(json.dumps(item) for item in sample_extracted_text)


@pytest.fixture
def mock_rust_backend(mocker):
    """Mock the Rust backend functions."""
    mock_extract = mocker.patch('pdf_backend_pdfium.extract_plain_text')
    mock_render = mocker.patch('pdf_backend_pdfium.render_page')
    mock_page_count = mocker.patch('pdf_backend_pdfium.page_count')
    mock_init_config = mocker.patch('pdf_backend_pdfium.init_config_with_profile')
    
    # Set up default return values
    mock_extract.return_value = [
        '{"page": 0, "text": "Test text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
    ]
    mock_render.return_value = (800, 600, b'fake_image_data' * 1000)
    mock_page_count.return_value = 2
    mock_init_config.return_value = None
    
    return {
        'extract_plain_text': mock_extract,
        'render_page': mock_render,
        'page_count': mock_page_count,
        'init_config_with_profile': mock_init_config
    }


@pytest.fixture
def mock_onnx_session(mocker):
    """Mock ONNX runtime session."""
    mock_session = mocker.patch('onnxruntime.InferenceSession')
    mock_instance = mocker.MagicMock()
    mock_session.return_value = mock_instance
    
    # Mock inference output
    mock_instance.run.return_value = [
        # Mock YOLO output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        [[0.1, 0.1, 0.5, 0.3, 0.9, 0.8, 0.1, 0.1, 0.0]]  # x, y, w, h, conf, class_probs...
    ]
    
    return mock_instance


@pytest.fixture
def skip_if_no_model():
    """Skip test if ONNX model file is not available."""
    model_path = Path("models/yolov8s-doclaynet.onnx")
    if not model_path.exists():
        pytest.skip(f"ONNX model not found at {model_path}")


@pytest.fixture
def skip_if_no_pdf():
    """Skip test if sample PDF is not available."""
    pdf_path = Path("sample.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}")


# Markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring ONNX model files"
    )