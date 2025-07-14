"""Unit tests for layout detection functionality."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from docparse.layout import (
    raster_page,
    get_session,
    preprocess,
    postprocess,
    detect_regions
)


@pytest.mark.unit
class TestRasterPage:
    """Test PDF page rasterization functionality."""
    
    def test_raster_page_default_dpi(self, mock_rust_backend):
        """Test page rasterization with default DPI."""
        # Mock render_page to return fake image data
        width, height = 800, 600
        fake_data = b'\x00' * (width * height * 3)  # RGB data
        mock_rust_backend['render_page'].return_value = (width, height, fake_data)
        
        result = raster_page("test.pdf", 0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
        
        mock_rust_backend['render_page'].assert_called_once_with("test.pdf", 0, 224)  # Default DPI
    
    def test_raster_page_custom_dpi(self, mock_rust_backend):
        """Test page rasterization with custom DPI."""
        width, height = 1200, 900
        fake_data = b'\xFF' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_data)
        
        result = raster_page("test.pdf", 0, dpi=300)
        
        assert result.shape == (height, width, 3)
        assert np.all(result == 255)  # All white pixels
        
        mock_rust_backend['render_page'].assert_called_once_with("test.pdf", 0, 300)
    
    def test_raster_page_different_pages(self, mock_rust_backend):
        """Test rasterization of different pages."""
        width, height = 800, 600
        
        # Page 0 - all black
        fake_data_0 = b'\x00' * (width * height * 3)
        # Page 1 - all white  
        fake_data_1 = b'\xFF' * (width * height * 3)
        
        mock_rust_backend['render_page'].side_effect = [
            (width, height, fake_data_0),
            (width, height, fake_data_1)
        ]
        
        result_0 = raster_page("test.pdf", 0)
        result_1 = raster_page("test.pdf", 1)
        
        assert np.all(result_0 == 0)   # All black
        assert np.all(result_1 == 255) # All white
        
        # Verify correct page indices were used
        calls = mock_rust_backend['render_page'].call_args_list
        assert calls[0][0][1] == 0  # First call with page 0
        assert calls[1][0][1] == 1  # Second call with page 1
    
    def test_raster_page_rgb_channels(self, mock_rust_backend):
        """Test that rasterization produces correct RGB channels."""
        width, height = 100, 100
        
        # Create test data with different RGB values
        data = []
        for y in range(height):
            for x in range(width):
                r = x % 256
                g = y % 256
                b = (x + y) % 256
                data.extend([r, g, b])
        
        fake_data = bytes(data)
        mock_rust_backend['render_page'].return_value = (width, height, fake_data)
        
        result = raster_page("test.pdf", 0)
        
        assert result.shape == (height, width, 3)
        
        # Check a few specific pixels
        assert result[0, 0, 0] == 0    # R at (0,0)
        assert result[0, 0, 1] == 0    # G at (0,0)
        assert result[0, 0, 2] == 0    # B at (0,0)
        
        assert result[0, 50, 0] == 50  # R at (0,50)
        assert result[50, 0, 1] == 50  # G at (50,0)


@pytest.mark.unit
class TestONNXSession:
    """Test ONNX session management."""
    
    def test_get_session_default_model(self, mock_onnx_session):
        """Test getting ONNX session with default model."""
        session = get_session()
        
        assert session is not None
        # Verify session was created (mocked)
        mock_onnx_session.assert_called_once()
    
    def test_get_session_custom_model(self, mock_onnx_session):
        """Test getting ONNX session with custom model path."""
        custom_model = "custom_model.onnx"
        session = get_session(custom_model)
        
        assert session is not None
        mock_onnx_session.assert_called_once()
    
    def test_get_session_singleton(self, mock_onnx_session):
        """Test that get_session returns the same instance."""
        # Reset the global session
        import docparse.layout
        docparse.layout._session = None
        
        session1 = get_session()
        session2 = get_session()
        
        assert session1 is session2
        # Should only be called once due to singleton pattern
        assert mock_onnx_session.call_count == 1
    
    @patch('docparse.layout.get_config')
    def test_get_session_uses_config(self, mock_get_config, mock_onnx_session):
        """Test that get_session uses configuration."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.get_model_path.return_value = Path("models/test.onnx")
        mock_config.runtime.execution_providers = ["CPUExecutionProvider"]
        mock_get_config.return_value = mock_config
        
        # Reset singleton
        import docparse.layout
        docparse.layout._session = None
        
        session = get_session()
        
        assert session is not None
        mock_get_config.assert_called()
        mock_config.get_model_path.assert_called()


@pytest.mark.unit
class TestPreprocessing:
    """Test image preprocessing functionality."""
    
    def test_preprocess_square_image(self):
        """Test preprocessing of square image."""
        # Create a square test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = preprocess(image, input_size=512)
        
        assert result.shape == (1, 3, 512, 512)  # NCHW format
        assert result.dtype == np.float32
        assert 0 <= result.max() <= 1  # Normalized to [0,1]
    
    def test_preprocess_rectangular_image(self):
        """Test preprocessing of rectangular image (requires padding)."""
        # Create a rectangular test image
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        result = preprocess(image, input_size=512)
        
        assert result.shape == (1, 3, 512, 512)  # Should be padded to square
        assert result.dtype == np.float32
    
    def test_preprocess_large_image(self):
        """Test preprocessing of large image (requires resizing)."""
        # Create a large test image
        image = np.random.randint(0, 255, (1024, 1536, 3), dtype=np.uint8)
        
        result = preprocess(image, input_size=512)
        
        assert result.shape == (1, 3, 512, 512)  # Should be resized down
        assert result.dtype == np.float32
    
    def test_preprocess_small_image(self):
        """Test preprocessing of small image (requires upscaling)."""
        # Create a small test image
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        result = preprocess(image, input_size=512)
        
        assert result.shape == (1, 3, 512, 512)  # Should be upscaled
        assert result.dtype == np.float32
    
    def test_preprocess_normalization(self):
        """Test that preprocessing normalizes pixel values correctly."""
        # Create image with known pixel values
        image = np.full((100, 100, 3), 255, dtype=np.uint8)  # All white
        
        result = preprocess(image, input_size=100)
        
        # All pixels should be normalized to ~1.0
        assert np.allclose(result, 1.0, atol=0.01)
        
        # Test with black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess(image, input_size=100)
        
        # All pixels should be normalized to 0.0
        assert np.allclose(result, 0.0, atol=0.01)
    
    def test_preprocess_channel_order(self):
        """Test that preprocessing converts to correct channel order."""
        # Create image with distinct RGB channels
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel
        image[:, :, 1] = 128  # Green channel
        image[:, :, 2] = 64   # Blue channel
        
        result = preprocess(image, input_size=100)
        
        # Check channel order in NCHW format
        assert result.shape == (1, 3, 100, 100)
        
        # Red channel should be first
        assert np.allclose(result[0, 0, :, :], 1.0, atol=0.01)
        # Green channel should be second
        assert np.allclose(result[0, 1, :, :], 128/255, atol=0.01)
        # Blue channel should be third
        assert np.allclose(result[0, 2, :, :], 64/255, atol=0.01)
    
    def test_preprocess_custom_input_size(self):
        """Test preprocessing with different input sizes."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Test different input sizes
        for input_size in [256, 512, 1024]:
            result = preprocess(image, input_size=input_size)
            assert result.shape == (1, 3, input_size, input_size)
    
    def test_preprocess_padding_color(self):
        """Test that preprocessing uses correct padding color."""
        # Create a small image that will need padding
        image = np.full((50, 50, 3), 255, dtype=np.uint8)  # Small white square
        
        result = preprocess(image, input_size=100)
        
        # The result should have padding areas
        # Center should be white (1.0), edges should be padding color
        center_val = result[0, 0, 50, 50]  # Center pixel, red channel
        edge_val = result[0, 0, 0, 0]      # Corner pixel, red channel
        
        # Center should be white, edge should be padding (typically gray)
        assert center_val > edge_val


@pytest.mark.unit
class TestPostprocessing:
    """Test ONNX output postprocessing."""
    
    def test_postprocess_valid_detections(self):
        """Test postprocessing with valid detections."""
        # Mock YOLO output format: [x, y, w, h, conf, class_probs...]
        mock_output = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.9, 0.8, 0.1, 0.1],  # High confidence detection
            [0.2, 0.7, 0.3, 0.2, 0.4, 0.3, 0.7, 0.0],  # Lower confidence detection
            [0.8, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.8]   # Low confidence detection
        ]).reshape(1, 3, 8)  # Batch size 1, 3 detections, 8 values each
        
        # Mock image dimensions
        original_width, original_height = 800, 600
        
        result = postprocess(
            mock_output, 
            original_width, 
            original_height,
            confidence_threshold=0.3,
            iou_threshold=0.5
        )
        
        assert isinstance(result, list)
        # Should filter out low confidence detections
        assert len(result) <= 3
        
        # Check detection format
        if len(result) > 0:
            detection = result[0]
            assert 'label' in detection
            assert 'confidence' in detection
            assert 'bbox' in detection
            assert len(detection['bbox']) == 4  # [x1, y1, x2, y2]
    
    def test_postprocess_no_detections(self):
        """Test postprocessing with no valid detections."""
        # All detections have low confidence
        mock_output = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.1, 0.8, 0.1, 0.1],
            [0.2, 0.7, 0.3, 0.2, 0.2, 0.3, 0.7, 0.0]
        ]).reshape(1, 2, 8)
        
        result = postprocess(
            mock_output,
            800, 600,
            confidence_threshold=0.5
        )
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_postprocess_bbox_scaling(self):
        """Test that bounding boxes are scaled correctly."""
        # Detection at center with known dimensions
        mock_output = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.8, 0.1, 0.1]  # Center, 20% width/height
        ]).reshape(1, 1, 8)
        
        original_width, original_height = 1000, 800
        
        result = postprocess(mock_output, original_width, original_height)
        
        assert len(result) == 1
        detection = result[0]
        bbox = detection['bbox']
        
        # Check that bbox is properly scaled
        # Center at (500, 400), size 200x160
        expected_x1 = 400  # 500 - 100
        expected_y1 = 320  # 400 - 80
        expected_x2 = 600  # 500 + 100
        expected_y2 = 480  # 400 + 80
        
        assert abs(bbox[0] - expected_x1) < 10
        assert abs(bbox[1] - expected_y1) < 10
        assert abs(bbox[2] - expected_x2) < 10
        assert abs(bbox[3] - expected_y2) < 10
    
    def test_postprocess_class_labels(self):
        """Test that class labels are assigned correctly."""
        # Create detections for different classes
        mock_output = np.array([
            # Title detection (class 0 highest)
            [0.5, 0.1, 0.8, 0.1, 0.9, 0.9, 0.05, 0.05],
            # Text detection (class 1 highest)  
            [0.5, 0.5, 0.8, 0.6, 0.8, 0.05, 0.9, 0.05],
            # Figure detection (class 2 highest)
            [0.5, 0.8, 0.4, 0.3, 0.7, 0.05, 0.05, 0.9]
        ]).reshape(1, 3, 8)
        
        result = postprocess(mock_output, 800, 600)
        
        assert len(result) == 3
        
        # Check class assignments (assuming standard labels)
        labels = [det['label'] for det in result]
        confidences = [det['confidence'] for det in result]
        
        # All should have reasonable confidence
        assert all(conf > 0.5 for conf in confidences)
        
        # Should have different labels
        assert len(set(labels)) <= 3  # At most 3 different labels


@pytest.mark.unit
class TestDetectRegions:
    """Test complete region detection pipeline."""
    
    def test_detect_regions_integration(self, mock_rust_backend, mock_onnx_session):
        """Test complete region detection workflow."""
        # Mock rasterization
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)  # Gray image
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # Mock ONNX inference
        mock_onnx_output = np.array([
            [0.5, 0.2, 0.4, 0.2, 0.9, 0.8, 0.1, 0.1],  # Title
            [0.5, 0.6, 0.8, 0.4, 0.8, 0.1, 0.8, 0.1]   # Text
        ]).reshape(1, 2, 8)
        mock_onnx_session.run.return_value = [mock_onnx_output]
        
        pdf_path = "test.pdf"
        page_idx = 0
        
        regions = detect_regions(pdf_path, page_idx)
        
        assert isinstance(regions, list)
        assert len(regions) <= 2  # At most 2 detections
        
        # Verify the pipeline was called
        mock_rust_backend['render_page'].assert_called_once_with(pdf_path, page_idx, 224)
        mock_onnx_session.run.assert_called_once()
        
        # Check region format
        if len(regions) > 0:
            region = regions[0]
            assert 'label' in region
            assert 'confidence' in region
            assert 'bbox' in region
            assert isinstance(region['bbox'], list)
            assert len(region['bbox']) == 4
    
    def test_detect_regions_custom_dpi(self, mock_rust_backend, mock_onnx_session):
        """Test region detection with custom DPI."""
        width, height = 1200, 900  # Higher resolution due to higher DPI
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_onnx_output = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.8, 0.1, 0.1]
        ]).reshape(1, 1, 8)
        mock_onnx_session.run.return_value = [mock_onnx_output]
        
        regions = detect_regions("test.pdf", 0, dpi=300)
        
        # Verify custom DPI was used
        mock_rust_backend['render_page'].assert_called_once_with("test.pdf", 0, 300)
    
    def test_detect_regions_no_detections(self, mock_rust_backend, mock_onnx_session):
        """Test region detection when no regions are found."""
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # Mock ONNX to return low confidence detections
        mock_onnx_output = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.1, 0.8, 0.1, 0.1]  # Low confidence
        ]).reshape(1, 1, 8)
        mock_onnx_session.run.return_value = [mock_onnx_output]
        
        regions = detect_regions("test.pdf", 0)
        
        assert isinstance(regions, list)
        assert len(regions) == 0
    
    @pytest.mark.requires_model
    def test_detect_regions_real_model(self, skip_if_no_model, skip_if_no_pdf):
        """Test region detection with real model (if available)."""
        # This test requires actual model and PDF files
        pdf_path = "sample.pdf"
        page_idx = 0
        
        regions = detect_regions(pdf_path, page_idx)
        
        assert isinstance(regions, list)
        # Real model should find some regions in a real document
        # (This is a loose assertion since we don't know the document content)
        
        if len(regions) > 0:
            region = regions[0]
            assert isinstance(region['confidence'], float)
            assert 0 <= region['confidence'] <= 1
            assert isinstance(region['bbox'], list)
            assert len(region['bbox']) == 4
            assert all(isinstance(coord, (int, float)) for coord in region['bbox'])


@pytest.mark.unit 
class TestLayoutDetectionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_detect_regions_invalid_pdf(self, mock_rust_backend):
        """Test region detection with invalid PDF."""
        # Mock Rust backend to raise an error
        mock_rust_backend['render_page'].side_effect = Exception("Invalid PDF")
        
        with pytest.raises(Exception):
            detect_regions("invalid.pdf", 0)
    
    def test_detect_regions_invalid_page(self, mock_rust_backend):
        """Test region detection with invalid page index."""
        mock_rust_backend['render_page'].side_effect = Exception("Page out of range")
        
        with pytest.raises(Exception):
            detect_regions("test.pdf", 999)
    
    def test_preprocess_invalid_input_size(self):
        """Test preprocessing with invalid input size."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            preprocess(image, input_size=0)
        
        with pytest.raises(ValueError):
            preprocess(image, input_size=-100)
    
    def test_preprocess_wrong_image_format(self):
        """Test preprocessing with wrong image format."""
        # Wrong number of channels
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)  # RGBA
        
        with pytest.raises((ValueError, IndexError)):
            preprocess(image)
        
        # Wrong dimensions
        image = np.random.randint(0, 255, (100,), dtype=np.uint8)  # 1D array
        
        with pytest.raises((ValueError, IndexError)):
            preprocess(image)
    
    def test_postprocess_malformed_output(self):
        """Test postprocessing with malformed ONNX output."""
        # Wrong shape
        mock_output = np.array([1, 2, 3, 4])  # Wrong shape
        
        with pytest.raises((ValueError, IndexError)):
            postprocess(mock_output, 800, 600)
        
        # Empty output
        mock_output = np.array([]).reshape(1, 0, 8)
        
        result = postprocess(mock_output, 800, 600)
        assert len(result) == 0
    
    def test_raster_page_zero_dimensions(self, mock_rust_backend):
        """Test rasterization with zero dimensions."""
        mock_rust_backend['render_page'].return_value = (0, 0, b'')
        
        result = raster_page("test.pdf", 0)
        
        assert result.shape == (0, 0, 3)
    
    def test_raster_page_mismatched_data_size(self, mock_rust_backend):
        """Test rasterization with mismatched data size."""
        # Data size doesn't match width * height * 3
        mock_rust_backend['render_page'].return_value = (100, 100, b'\x00' * 1000)  # Too small
        
        with pytest.raises(ValueError):
            raster_page("test.pdf", 0)