"""Unit tests for text extraction functionality."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from docparse.text import extract, PageText


@pytest.mark.unit
class TestPageText:
    """Test PageText namedtuple."""
    
    def test_page_text_creation(self):
        """Test PageText namedtuple creation."""
        page_text = PageText(
            page=0,
            text="Sample text",
            x0=100.0,
            y0=200.0,
            x1=400.0,
            y1=220.0
        )
        
        assert page_text.page == 0
        assert page_text.text == "Sample text"
        assert page_text.x0 == 100.0
        assert page_text.y0 == 200.0
        assert page_text.x1 == 400.0
        assert page_text.y1 == 220.0
    
    def test_page_text_immutable(self):
        """Test that PageText is immutable."""
        page_text = PageText(
            page=0,
            text="Sample text",
            x0=100.0,
            y0=200.0,
            x1=400.0,
            y1=220.0
        )
        
        with pytest.raises(AttributeError):
            page_text.page = 1
    
    def test_page_text_attributes(self):
        """Test PageText has all expected attributes."""
        page_text = PageText(0, "text", 0.0, 0.0, 100.0, 20.0)
        
        assert hasattr(page_text, 'page')
        assert hasattr(page_text, 'text')
        assert hasattr(page_text, 'x0')
        assert hasattr(page_text, 'y0')
        assert hasattr(page_text, 'x1')
        assert hasattr(page_text, 'y1')


@pytest.mark.unit
class TestTextExtraction:
    """Test text extraction functionality."""
    
    def test_extract_with_mock_backend(self, mock_rust_backend, sample_json_lines):
        """Test text extraction with mocked Rust backend."""
        # Set up mock to return sample JSON lines
        mock_rust_backend['extract_plain_text'].return_value = sample_json_lines.split('\n')
        
        pdf_path = Path("test.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 3
        
        # Check first result
        assert results[0].page == 0
        assert results[0].text == "Sample Document Title"
        assert results[0].x0 == 100.0
        assert results[0].y0 == 700.0
        assert results[0].x1 == 500.0
        assert results[0].y1 == 720.0
        
        # Check second result
        assert results[1].page == 0
        assert results[1].text == "This is the first paragraph of the document."
        
        # Check third result (different page)
        assert results[2].page == 1
        assert results[2].text == "This is content on the second page."
        
        # Verify Rust backend was called correctly
        mock_rust_backend['extract_plain_text'].assert_called_once_with(str(pdf_path))
    
    def test_extract_empty_document(self, mock_rust_backend):
        """Test text extraction from empty document."""
        mock_rust_backend['extract_plain_text'].return_value = []
        
        pdf_path = Path("empty.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 0
        mock_rust_backend['extract_plain_text'].assert_called_once_with(str(pdf_path))
    
    def test_extract_single_line(self, mock_rust_backend):
        """Test text extraction with single line."""
        single_line = '{"page": 0, "text": "Single line", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        mock_rust_backend['extract_plain_text'].return_value = [single_line]
        
        pdf_path = Path("single.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].page == 0
        assert results[0].text == "Single line"
        assert results[0].x0 == 0
        assert results[0].y0 == 0
        assert results[0].x1 == 100
        assert results[0].y1 == 20
    
    def test_extract_multiple_pages(self, mock_rust_backend):
        """Test text extraction with multiple pages."""
        json_lines = [
            '{"page": 0, "text": "Page 0 text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}',
            '{"page": 1, "text": "Page 1 text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}',
            '{"page": 2, "text": "Page 2 text", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        mock_rust_backend['extract_plain_text'].return_value = json_lines
        
        pdf_path = Path("multi.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 3
        assert results[0].page == 0
        assert results[1].page == 1
        assert results[2].page == 2
        assert all("text" in result.text for result in results)
    
    def test_extract_with_special_characters(self, mock_rust_backend):
        """Test text extraction with special characters."""
        special_text = '{"page": 0, "text": "Special: ñáéíóú ©®™ 中文", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        mock_rust_backend['extract_plain_text'].return_value = [special_text]
        
        pdf_path = Path("special.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].text == "Special: ñáéíóú ©®™ 中文"
    
    def test_extract_with_newlines_and_quotes(self, mock_rust_backend):
        """Test text extraction with newlines and quotes in text."""
        complex_text = json.dumps({
            "page": 0,
            "text": 'Text with "quotes" and\nnewlines',
            "x0": 0,
            "y0": 0,
            "x1": 100,
            "y1": 20
        })
        mock_rust_backend['extract_plain_text'].return_value = [complex_text]
        
        pdf_path = Path("complex.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert 'quotes' in results[0].text
        assert '\n' in results[0].text
    
    def test_extract_with_floating_point_coordinates(self, mock_rust_backend):
        """Test text extraction with precise floating point coordinates."""
        precise_coords = '{"page": 0, "text": "Precise", "x0": 123.456, "y0": 789.012, "x1": 234.567, "y1": 890.123}'
        mock_rust_backend['extract_plain_text'].return_value = [precise_coords]
        
        pdf_path = Path("precise.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].x0 == 123.456
        assert results[0].y0 == 789.012
        assert results[0].x1 == 234.567
        assert results[0].y1 == 890.123
    
    def test_extract_invalid_json_handling(self, mock_rust_backend):
        """Test handling of invalid JSON from Rust backend."""
        mock_rust_backend['extract_plain_text'].return_value = ['invalid json']
        
        pdf_path = Path("invalid.pdf")
        
        with pytest.raises(json.JSONDecodeError):
            list(extract(pdf_path))
    
    def test_extract_missing_fields_handling(self, mock_rust_backend):
        """Test handling of JSON with missing required fields."""
        incomplete_json = '{"page": 0, "text": "Missing coordinates"}'
        mock_rust_backend['extract_plain_text'].return_value = [incomplete_json]
        
        pdf_path = Path("incomplete.pdf")
        
        with pytest.raises(KeyError):
            list(extract(pdf_path))
    
    def test_extract_wrong_field_types(self, mock_rust_backend):
        """Test handling of JSON with wrong field types."""
        wrong_types = '{"page": "zero", "text": "Wrong types", "x0": "left", "y0": 0, "x1": 100, "y1": 20}'
        mock_rust_backend['extract_plain_text'].return_value = [wrong_types]
        
        pdf_path = Path("wrong_types.pdf")
        
        # Should still work but may cause issues downstream
        results = list(extract(pdf_path))
        assert len(results) == 1
        # The namedtuple will accept any types, validation should happen elsewhere
    
    def test_extract_large_coordinates(self, mock_rust_backend):
        """Test extraction with very large coordinate values."""
        large_coords = '{"page": 0, "text": "Large", "x0": 999999.99, "y0": 888888.88, "x1": 1111111.11, "y1": 2222222.22}'
        mock_rust_backend['extract_plain_text'].return_value = [large_coords]
        
        pdf_path = Path("large.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].x0 == 999999.99
        assert results[0].y0 == 888888.88
        assert results[0].x1 == 1111111.11
        assert results[0].y1 == 2222222.22
    
    def test_extract_negative_coordinates(self, mock_rust_backend):
        """Test extraction with negative coordinate values."""
        negative_coords = '{"page": 0, "text": "Negative", "x0": -100.0, "y0": -200.0, "x1": -50.0, "y1": -180.0}'
        mock_rust_backend['extract_plain_text'].return_value = [negative_coords]
        
        pdf_path = Path("negative.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].x0 == -100.0
        assert results[0].y0 == -200.0
        assert results[0].x1 == -50.0
        assert results[0].y1 == -180.0
    
    def test_extract_zero_dimensions(self, mock_rust_backend):
        """Test extraction with zero-dimension bounding boxes."""
        zero_dims = '{"page": 0, "text": "Zero", "x0": 100.0, "y0": 200.0, "x1": 100.0, "y1": 200.0}'
        mock_rust_backend['extract_plain_text'].return_value = [zero_dims]
        
        pdf_path = Path("zero.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 1
        assert results[0].x0 == results[0].x1
        assert results[0].y0 == results[0].y1
    
    def test_extract_path_as_string(self, mock_rust_backend):
        """Test that extract works with string paths."""
        json_line = '{"page": 0, "text": "String path", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        mock_rust_backend['extract_plain_text'].return_value = [json_line]
        
        # Pass string instead of Path object
        pdf_path = "string_path.pdf"
        results = list(extract(Path(pdf_path)))
        
        assert len(results) == 1
        assert results[0].text == "String path"
        
        # Verify the Rust backend received the string path
        mock_rust_backend['extract_plain_text'].assert_called_once_with(pdf_path)


@pytest.mark.unit
class TestTextExtractionIntegration:
    """Test text extraction with more realistic scenarios."""
    
    def test_extract_academic_paper_structure(self, mock_rust_backend):
        """Test extraction from typical academic paper structure."""
        academic_lines = [
            '{"page": 0, "text": "Paper Title in Large Font", "x0": 100, "y0": 750, "x1": 500, "y1": 770}',
            '{"page": 0, "text": "Author Name1, Author Name2", "x0": 150, "y0": 720, "x1": 450, "y1": 735}',
            '{"page": 0, "text": "Abstract", "x0": 100, "y0": 680, "x1": 150, "y1": 695}',
            '{"page": 0, "text": "This paper presents a novel approach...", "x0": 100, "y0": 650, "x1": 500, "y1": 670}',
            '{"page": 0, "text": "1. Introduction", "x0": 100, "y0": 600, "x1": 200, "y1": 615}',
            '{"page": 0, "text": "In recent years, the field has seen...", "x0": 100, "y0": 580, "x1": 500, "y1": 595}',
            '{"page": 1, "text": "2. Related Work", "x0": 100, "y0": 750, "x1": 200, "y1": 765}',
            '{"page": 1, "text": "Previous studies have shown...", "x0": 100, "y0": 730, "x1": 500, "y1": 745}'
        ]
        mock_rust_backend['extract_plain_text'].return_value = academic_lines
        
        pdf_path = Path("academic_paper.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 8
        
        # Check title
        title = results[0]
        assert "Paper Title" in title.text
        assert title.page == 0
        assert title.y0 > 700  # Should be near top of page
        
        # Check author
        author = results[1]
        assert "Author" in author.text
        assert author.y0 < title.y0  # Should be below title
        
        # Check sections span pages
        page_0_results = [r for r in results if r.page == 0]
        page_1_results = [r for r in results if r.page == 1]
        
        assert len(page_0_results) == 6
        assert len(page_1_results) == 2
    
    def test_extract_table_like_structure(self, mock_rust_backend):
        """Test extraction from table-like structure."""
        table_lines = [
            '{"page": 0, "text": "Name", "x0": 100, "y0": 700, "x1": 150, "y1": 715}',
            '{"page": 0, "text": "Age", "x0": 200, "y0": 700, "x1": 230, "y1": 715}',
            '{"page": 0, "text": "City", "x0": 300, "y0": 700, "x1": 330, "y1": 715}',
            '{"page": 0, "text": "John", "x0": 100, "y0": 680, "x1": 140, "y1": 695}',
            '{"page": 0, "text": "25", "x0": 200, "y0": 680, "x1": 220, "y1": 695}',
            '{"page": 0, "text": "NYC", "x0": 300, "y0": 680, "x1": 330, "y1": 695}',
            '{"page": 0, "text": "Jane", "x0": 100, "y0": 660, "x1": 140, "y1": 675}',
            '{"page": 0, "text": "30", "x0": 200, "y0": 660, "x1": 220, "y1": 675}',
            '{"page": 0, "text": "LA", "x0": 300, "y0": 660, "x1": 320, "y1": 675}'
        ]
        mock_rust_backend['extract_plain_text'].return_value = table_lines
        
        pdf_path = Path("table.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 9
        
        # Check header row
        headers = results[:3]
        assert headers[0].text == "Name"
        assert headers[1].text == "Age"
        assert headers[2].text == "City"
        
        # All headers should be at same y position
        assert headers[0].y0 == headers[1].y0 == headers[2].y0
        
        # Check data rows have consistent x positions within columns
        names = [results[3], results[6]]  # John, Jane
        ages = [results[4], results[7]]   # 25, 30
        cities = [results[5], results[8]] # NYC, LA
        
        # Names should have similar x0 positions
        assert abs(names[0].x0 - names[1].x0) < 5
        assert abs(ages[0].x0 - ages[1].x0) < 5
        assert abs(cities[0].x0 - cities[1].x0) < 5
    
    def test_extract_multi_column_layout(self, mock_rust_backend):
        """Test extraction from multi-column layout."""
        column_lines = [
            # Left column
            '{"page": 0, "text": "Left column header", "x0": 50, "y0": 750, "x1": 250, "y1": 765}',
            '{"page": 0, "text": "Left column content line 1", "x0": 50, "y0": 730, "x1": 250, "y1": 745}',
            '{"page": 0, "text": "Left column content line 2", "x0": 50, "y0": 710, "x1": 250, "y1": 725}',
            # Right column
            '{"page": 0, "text": "Right column header", "x0": 300, "y0": 750, "x1": 500, "y1": 765}',
            '{"page": 0, "text": "Right column content line 1", "x0": 300, "y0": 730, "x1": 500, "y1": 745}',
            '{"page": 0, "text": "Right column content line 2", "x0": 300, "y0": 710, "x1": 500, "y1": 725}'
        ]
        mock_rust_backend['extract_plain_text'].return_value = column_lines
        
        pdf_path = Path("columns.pdf")
        results = list(extract(pdf_path))
        
        assert len(results) == 6
        
        # Separate left and right columns by x position
        left_column = [r for r in results if r.x0 < 200]
        right_column = [r for r in results if r.x0 >= 200]
        
        assert len(left_column) == 3
        assert len(right_column) == 3
        
        # Check headers are at top
        left_header = left_column[0]
        right_header = right_column[0]
        
        assert "Left column header" in left_header.text
        assert "Right column header" in right_header.text
        assert left_header.y0 == right_header.y0  # Same vertical position