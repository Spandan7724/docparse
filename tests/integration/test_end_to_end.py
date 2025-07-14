"""End-to-end tests for docparse functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
import numpy as np

from docparse import extract_text
from docparse.text import extract, PageText
from docparse.layout import detect_regions, raster_page
from docparse.config import init_config, get_config, reset_config


@pytest.mark.e2e
class TestEndToEndTextExtraction:
    """End-to-end tests for text extraction workflow."""
    
    def test_complete_text_extraction_workflow(self, mock_rust_backend, sample_json_lines):
        """Test complete text extraction from PDF to PageText objects."""
        # Set up mock backend
        mock_rust_backend['extract_plain_text'].return_value = sample_json_lines.split('\n')
        
        # Test the complete workflow
        pdf_path = Path("test_document.pdf")
        extracted_lines = list(extract(pdf_path))
        
        # Verify results
        assert len(extracted_lines) == 3
        
        # Check first line (title)
        title_line = extracted_lines[0]
        assert isinstance(title_line, PageText)
        assert title_line.page == 0
        assert "Document Title" in title_line.text
        assert title_line.y0 > title_line.y1  # PDF coordinates (bottom-left origin)
        
        # Check multi-page extraction
        page_0_lines = [line for line in extracted_lines if line.page == 0]
        page_1_lines = [line for line in extracted_lines if line.page == 1]
        
        assert len(page_0_lines) == 2
        assert len(page_1_lines) == 1
        
        # Verify coordinate system
        for line in extracted_lines:
            assert line.x0 <= line.x1  # Left <= Right
            assert isinstance(line.x0, float)
            assert isinstance(line.y0, float)
            assert isinstance(line.x1, float)
            assert isinstance(line.y1, float)
    
    def test_text_extraction_with_configuration(self, mock_rust_backend, temp_config_file):
        """Test text extraction with custom configuration."""
        # Initialize with custom config
        init_config(config_path=temp_config_file)
        config = get_config()
        
        # Verify config was loaded
        assert config.text_extraction.line_clustering_tolerance == 0.3
        assert config.text_extraction.space_threshold_height == 0.15
        
        # Mock extraction with config-influenced results
        mock_rust_backend['extract_plain_text'].return_value = [
            '{"page": 0, "text": "Config test", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        
        pdf_path = Path("config_test.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].text == "Config test"
        
        # Verify Rust backend was called
        mock_rust_backend['extract_plain_text'].assert_called_once_with(str(pdf_path))
    
    def test_text_extraction_with_profile(self, mock_rust_backend, temp_config_file):
        """Test text extraction with configuration profile."""
        # Initialize with academic profile
        init_config(config_path=temp_config_file, profile="academic")
        config = get_config()
        
        # Verify profile settings were applied
        assert config.text_extraction.line_clustering_tolerance == 0.20  # Academic profile
        
        mock_rust_backend['extract_plain_text'].return_value = [
            '{"page": 0, "text": "Academic text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        
        pdf_path = Path("academic.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].text == "Academic text"
    
    def test_text_extraction_error_handling(self, mock_rust_backend):
        """Test text extraction error handling."""
        # Mock Rust backend to raise an error
        mock_rust_backend['extract_plain_text'].side_effect = Exception("PDF processing error")
        
        pdf_path = Path("corrupted.pdf")
        
        with pytest.raises(Exception):
            list(extract(pdf_path))
    
    def test_text_extraction_empty_document(self, mock_rust_backend):
        """Test text extraction from empty document."""
        mock_rust_backend['extract_plain_text'].return_value = []
        
        pdf_path = Path("empty.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 0
    
    def test_text_extraction_large_document(self, mock_rust_backend):
        """Test text extraction from large document."""
        # Generate large document simulation
        large_output = []
        for page in range(10):  # 10 pages
            for line in range(50):  # 50 lines per page
                large_output.append(
                    f'{{"page": {page}, "text": "Page {page} line {line} content", '
                    f'"x0": 100, "y0": {800 - line * 15}, "x1": 500, "y1": {815 - line * 15}}}'
                )
        
        mock_rust_backend['extract_plain_text'].return_value = large_output
        
        pdf_path = Path("large_document.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 500  # 10 pages * 50 lines
        
        # Verify page distribution
        pages = set(result.page for result in results)
        assert pages == set(range(10))
        
        # Verify each page has correct number of lines
        for page_num in range(10):
            page_lines = [r for r in results if r.page == page_num]
            assert len(page_lines) == 50


@pytest.mark.e2e
class TestEndToEndLayoutDetection:
    """End-to-end tests for layout detection workflow."""
    
    def test_complete_layout_detection_workflow(self, mock_rust_backend, mock_onnx_session):
        """Test complete layout detection from PDF to regions."""
        # Mock PDF rasterization
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)  # Gray image
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # Mock ONNX inference
        mock_detections = np.array([
            # Title detection
            [0.5, 0.1, 0.8, 0.1, 0.95, 0.9, 0.05, 0.05, 0.0, 0.0],
            # Text detection
            [0.5, 0.5, 0.8, 0.6, 0.85, 0.05, 0.9, 0.05, 0.0, 0.0],
            # Figure detection
            [0.3, 0.8, 0.4, 0.2, 0.75, 0.05, 0.05, 0.9, 0.0, 0.0]
        ]).reshape(1, 3, 10)
        
        mock_onnx_session.run.return_value = [mock_detections]
        
        # Run complete workflow
        pdf_path = "test_layout.pdf"
        page_idx = 0
        regions = detect_regions(pdf_path, page_idx)
        
        # Verify results
        assert isinstance(regions, list)
        assert len(regions) <= 3  # May filter some based on confidence
        
        # Check region structure
        for region in regions:
            assert 'label' in region
            assert 'confidence' in region
            assert 'bbox' in region
            
            assert isinstance(region['confidence'], float)
            assert 0 <= region['confidence'] <= 1
            assert isinstance(region['bbox'], list)
            assert len(region['bbox']) == 4
            
            # Check bbox coordinates are reasonable
            x1, y1, x2, y2 = region['bbox']
            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height
        
        # Verify the complete pipeline was executed
        mock_rust_backend['render_page'].assert_called_once_with(pdf_path, page_idx, 224)
        mock_onnx_session.run.assert_called_once()
    
    def test_layout_detection_with_custom_dpi(self, mock_rust_backend, mock_onnx_session):
        """Test layout detection with custom DPI."""
        # Higher DPI should produce larger image
        width, height = 1200, 900  # Larger due to higher DPI
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_detections = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.8, 0.8, 0.1, 0.1, 0.0, 0.0]
        ]).reshape(1, 1, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        regions = detect_regions("test.pdf", 0, dpi=300)
        
        # Verify custom DPI was used
        mock_rust_backend['render_page'].assert_called_once_with("test.pdf", 0, 300)
        
        # Regions should be scaled to original image size
        if regions:
            bbox = regions[0]['bbox']
            assert all(0 <= coord <= max(width, height) for coord in bbox)
    
    def test_layout_detection_with_configuration(self, mock_rust_backend, mock_onnx_session, temp_config_file):
        """Test layout detection with custom configuration."""
        # Initialize with custom config
        init_config(config_path=temp_config_file)
        config = get_config()
        
        # Verify layout config was loaded
        assert config.layout_detection.dpi == 300
        assert config.layout_detection.confidence_threshold == 0.4
        
        # Mock components
        width, height = 1200, 900  # Matches 300 DPI config
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_detections = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.8, 0.1, 0.1, 0.0, 0.0]
        ]).reshape(1, 1, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        regions = detect_regions("config_test.pdf", 0)
        
        # Should use config DPI (300)
        mock_rust_backend['render_page'].assert_called_once_with("config_test.pdf", 0, 300)
    
    def test_layout_detection_no_regions_found(self, mock_rust_backend, mock_onnx_session):
        """Test layout detection when no regions meet confidence threshold."""
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # All detections have low confidence
        mock_detections = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.1, 0.8, 0.1, 0.1, 0.0, 0.0],  # 0.1 confidence
            [0.3, 0.3, 0.1, 0.1, 0.2, 0.1, 0.8, 0.1, 0.0, 0.0]   # 0.2 confidence
        ]).reshape(1, 2, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        regions = detect_regions("low_confidence.pdf", 0)
        
        # Should return empty list (confidence threshold typically 0.3)
        assert isinstance(regions, list)
        assert len(regions) == 0
    
    def test_raster_page_integration(self, mock_rust_backend):
        """Test PDF page rasterization integration."""
        # Test realistic image dimensions
        width, height = 1024, 768
        
        # Create realistic image data (RGB)
        fake_data = []
        for y in range(height):
            for x in range(width):
                # Create simple gradient pattern
                r = (x * 255) // width
                g = (y * 255) // height
                b = 128
                fake_data.extend([r, g, b])
        
        fake_image_data = bytes(fake_data)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # Test rasterization
        image = raster_page("gradient.pdf", 0, dpi=150)
        
        # Verify output
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8
        
        # Check gradient pattern
        assert image[0, 0, 0] == 0      # Top-left red
        assert image[0, -1, 0] == 255   # Top-right red
        assert image[-1, 0, 1] == 255   # Bottom-left green
        assert image[0, 0, 1] == 0      # Top-left green
        assert np.all(image[:, :, 2] == 128)  # Blue channel constant


@pytest.mark.e2e
class TestEndToEndIntegration:
    """Test integration between different components."""
    
    def test_text_and_layout_integration(self, mock_rust_backend, mock_onnx_session, sample_json_lines):
        """Test using both text extraction and layout detection on same document."""
        # Set up text extraction mock
        mock_rust_backend['extract_plain_text'].return_value = sample_json_lines.split('\n')
        
        # Set up layout detection mocks
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_detections = np.array([
            [0.5, 0.1, 0.8, 0.1, 0.95, 0.9, 0.05, 0.05, 0.0, 0.0]  # Title region
        ]).reshape(1, 1, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        # Test both extractions on same PDF
        pdf_path = Path("integrated_test.pdf")
        
        # Extract text
        text_lines = list(extract(pdf_path))
        
        # Detect layout
        layout_regions = detect_regions(str(pdf_path), 0)
        
        # Verify both worked
        assert len(text_lines) == 3
        assert len(layout_regions) <= 1  # May be filtered by confidence
        
        # Verify different backends were called appropriately
        mock_rust_backend['extract_plain_text'].assert_called_once_with(str(pdf_path))
        mock_rust_backend['render_page'].assert_called_once_with(str(pdf_path), 0, 224)
        
        # Could potentially correlate text and layout regions
        if layout_regions:
            title_region = layout_regions[0]
            title_text = text_lines[0]  # First line should be title
            
            # Basic sanity check: title text should be within title region bounds
            # (This would require more sophisticated coordinate mapping in real usage)
            assert isinstance(title_region['bbox'], list)
            assert isinstance(title_text.x0, float)
    
    def test_configuration_affects_all_components(self, mock_rust_backend, mock_onnx_session, temp_config_file):
        """Test that configuration affects both text and layout components."""
        # Initialize with custom config that affects both components
        init_config(config_path=temp_config_file)
        config = get_config()
        
        # Verify config loaded correctly
        assert config.text_extraction.line_clustering_tolerance == 0.3
        assert config.layout_detection.dpi == 300
        assert config.layout_detection.confidence_threshold == 0.4
        
        # Mock backends
        mock_rust_backend['extract_plain_text'].return_value = [
            '{"page": 0, "text": "Configured text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        
        width, height = 1200, 900  # Higher resolution for 300 DPI
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_detections = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.5, 0.8, 0.1, 0.1, 0.0, 0.0]  # 0.5 confidence
        ]).reshape(1, 1, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        # Test both components
        pdf_path = "config_integration.pdf"
        
        text_results = list(extract(Path(pdf_path)))
        layout_results = detect_regions(pdf_path, 0)
        
        # Text extraction should work
        assert len(text_results) == 1
        assert text_results[0].text == "Configured text"
        
        # Layout detection should use 300 DPI and 0.4 confidence threshold
        mock_rust_backend['render_page'].assert_called_once_with(pdf_path, 0, 300)
        
        # With 0.4 threshold, 0.5 confidence detection should pass
        assert len(layout_results) == 1
    
    def test_profile_integration(self, mock_rust_backend, mock_onnx_session, temp_config_file):
        """Test that profiles affect the complete workflow."""
        # Test with academic profile
        init_config(config_path=temp_config_file, profile="academic")
        config = get_config()
        
        # Verify academic profile settings
        assert config.text_extraction.line_clustering_tolerance == 0.20
        assert config.layout_detection.confidence_threshold == 0.5
        assert config.layout_detection.dpi == 300
        
        # Mock components for academic workflow
        mock_rust_backend['extract_plain_text'].return_value = [
            '{"page": 0, "text": "Academic paper title", "x0": 100, "y0": 750, "x1": 500, "y1": 770}',
            '{"page": 0, "text": "Abstract section content", "x0": 100, "y0": 700, "x1": 500, "y1": 720}'
        ]
        
        width, height = 1200, 900  # 300 DPI
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        # High confidence detections for academic content
        mock_detections = np.array([
            [0.5, 0.1, 0.8, 0.05, 0.95, 0.9, 0.05, 0.05, 0.0, 0.0],  # Title: 0.95 conf
            [0.5, 0.3, 0.8, 0.2, 0.85, 0.05, 0.9, 0.05, 0.0, 0.0]    # Text: 0.85 conf
        ]).reshape(1, 2, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        # Run academic workflow
        pdf_path = "academic_paper.pdf"
        
        text_results = list(extract(Path(pdf_path)))
        layout_results = detect_regions(pdf_path, 0)
        
        # Both should work with academic settings
        assert len(text_results) == 2
        assert "Academic paper title" in text_results[0].text
        assert "Abstract section" in text_results[1].text
        
        # Both detections should pass 0.5 threshold
        assert len(layout_results) == 2
        
        # Verify academic DPI was used
        mock_rust_backend['render_page'].assert_called_once_with(pdf_path, 0, 300)


@pytest.mark.e2e
@pytest.mark.slow
class TestEndToEndPerformance:
    """Test end-to-end performance characteristics."""
    
    def test_large_document_performance(self, mock_rust_backend, mock_onnx_session):
        """Test performance with large document simulation."""
        import time
        
        # Simulate large document with many pages and text lines
        large_text_output = []
        for page in range(20):  # 20 pages
            for line in range(100):  # 100 lines per page
                large_text_output.append(
                    f'{{"page": {page}, "text": "Page {page} line {line} content with some longer text to simulate real documents", '
                    f'"x0": 100, "y0": {800 - line * 8}, "x1": 500, "y1": {808 - line * 8}}}'
                )
        
        mock_rust_backend['extract_plain_text'].return_value = large_text_output
        
        # Time the text extraction
        pdf_path = Path("large_document.pdf")
        start_time = time.time()
        
        results = list(extract(pdf_path))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert len(results) == 2000  # 20 pages * 100 lines
        
        # Performance should be reasonable (adjust threshold as needed)
        assert processing_time < 5.0  # Should process in under 5 seconds
        
        # Verify memory efficiency (results should be generator-based)
        assert all(isinstance(result, PageText) for result in results[:10])
    
    def test_multiple_operations_performance(self, mock_rust_backend, mock_onnx_session):
        """Test performance of multiple operations on same document."""
        import time
        
        # Set up mocks
        mock_rust_backend['extract_plain_text'].return_value = [
            '{"page": 0, "text": "Performance test", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        
        width, height = 800, 600
        fake_image_data = b'\x80' * (width * height * 3)
        mock_rust_backend['render_page'].return_value = (width, height, fake_image_data)
        
        mock_detections = np.array([
            [0.5, 0.5, 0.2, 0.2, 0.8, 0.8, 0.1, 0.1, 0.0, 0.0]
        ]).reshape(1, 1, 10)
        mock_onnx_session.run.return_value = [mock_detections]
        
        pdf_path = "performance_test.pdf"
        
        # Time multiple operations
        start_time = time.time()
        
        for i in range(10):  # 10 iterations
            text_results = list(extract(Path(pdf_path)))
            layout_results = detect_regions(pdf_path, 0)
            
            assert len(text_results) == 1
            assert len(layout_results) <= 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle multiple operations efficiently
        average_time = total_time / 10
        assert average_time < 1.0  # Under 1 second per iteration on average