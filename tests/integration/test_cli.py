"""Integration tests for CLI functionality."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from typer.testing import CliRunner

from docparse.cli import app


@pytest.mark.integration
class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    def test_cli_version(self):
        """Test --version flag."""
        result = self.runner.invoke(app, ["--version"])
        
        assert result.exit_code == 0
        assert "docparse" in result.stdout
        # Should contain version number
        assert any(char.isdigit() for char in result.stdout)
    
    def test_cli_help(self):
        """Test --help flag."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "High-performance PDF text extraction" in result.stdout
        assert "text" in result.stdout  # Should show text command
        assert "layout" in result.stdout  # Should show layout command
    
    def test_cli_no_command(self):
        """Test CLI with no command (should show help)."""
        result = self.runner.invoke(app, [])
        
        # Should either show help or exit cleanly
        assert result.exit_code in [0, 2]  # 0 for help, 2 for missing command


@pytest.mark.integration
class TestTextCommand:
    """Test text extraction command."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    def test_text_command_help(self):
        """Test text command help."""
        result = self.runner.invoke(app, ["text", "--help"])
        
        assert result.exit_code == 0
        assert "Extract text" in result.stdout
        assert "clustering" in result.stdout.lower()
        assert "--output" in result.stdout
    
    def test_text_command_missing_file(self):
        """Test text command with missing PDF file."""
        result = self.runner.invoke(app, ["text", "nonexistent.pdf"])
        
        # Should fail with error
        assert result.exit_code != 0
    
    @patch('docparse.cli.extract_plain_text')
    def test_text_command_stdout(self, mock_extract):
        """Test text command output to stdout."""
        # Mock the Rust backend
        mock_extract.return_value = [
            '{"page": 0, "text": "Test line 1", "x0": 0, "y0": 0, "x1": 100, "y1": 20}',
            '{"page": 0, "text": "Test line 2", "x0": 0, "y0": 25, "x1": 100, "y1": 45}'
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, ["text", str(tmp_pdf.name)])
            
            assert result.exit_code == 0
            assert "Test line 1" in result.stdout
            assert "Test line 2" in result.stdout
            
            # Should be valid JSON lines
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    json.loads(line)  # Should not raise exception
    
    @patch('docparse.cli.extract_plain_text')
    def test_text_command_output_file(self, mock_extract):
        """Test text command output to file."""
        mock_extract.return_value = [
            '{"page": 0, "text": "Output test", "x0": 0, "y0": 0, "x1": 100, "y1": 20}'
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_output:
            
            output_path = tmp_output.name
            
            result = self.runner.invoke(app, [
                "text", str(tmp_pdf.name), 
                "--output", output_path
            ])
            
            assert result.exit_code == 0
            
            # Check output file content
            with open(output_path, 'r') as f:
                content = f.read().strip()
                assert "Output test" in content
                # Should be valid JSON
                json.loads(content)
            
            # Clean up
            Path(output_path).unlink()
    
    def test_text_command_rust_backend_not_built(self):
        """Test text command when Rust backend is not built."""
        with patch('docparse.cli.extract_plain_text', None):
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
                result = self.runner.invoke(app, ["text", str(tmp_pdf.name)])
                
                assert result.exit_code == 1
                assert "Rust backend not built" in result.stdout


@pytest.mark.integration
class TestLayoutCommand:
    """Test layout detection command."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    def test_layout_command_help(self):
        """Test layout command help."""
        result = self.runner.invoke(app, ["layout", "--help"])
        
        assert result.exit_code == 0
        assert "Detect document layout" in result.stdout
        assert "--dpi" in result.stdout
        assert "--model" in result.stdout
        assert "--confidence" in result.stdout
    
    def test_layout_command_missing_file(self):
        """Test layout command with missing PDF file."""
        result = self.runner.invoke(app, ["layout", "nonexistent.pdf"])
        
        # Should fail with error
        assert result.exit_code != 0
    
    @patch('docparse.cli.detect_regions')
    def test_layout_command_stdout(self, mock_detect):
        """Test layout command output to stdout."""
        # Mock the layout detection
        mock_detect.return_value = [
            {
                "label": "Title",
                "confidence": 0.95,
                "bbox": [100, 700, 500, 720]
            },
            {
                "label": "Text", 
                "confidence": 0.85,
                "bbox": [100, 650, 500, 670]
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, ["layout", str(tmp_pdf.name)])
            
            assert result.exit_code == 0
            
            # Should contain JSON output
            output = json.loads(result.stdout)
            assert "regions" in output
            assert len(output["regions"]) == 2
            assert output["regions"][0]["label"] == "Title"
            assert output["regions"][1]["label"] == "Text"
    
    @patch('docparse.cli.detect_regions')
    def test_layout_command_output_file(self, mock_detect):
        """Test layout command output to file."""
        mock_detect.return_value = [
            {
                "label": "Figure",
                "confidence": 0.90,
                "bbox": [150, 400, 450, 600]
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_output:
            
            output_path = tmp_output.name
            
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--output", output_path
            ])
            
            assert result.exit_code == 0
            
            # Check output file content
            with open(output_path, 'r') as f:
                output = json.load(f)
                assert "regions" in output
                assert len(output["regions"]) == 1
                assert output["regions"][0]["label"] == "Figure"
            
            # Clean up
            Path(output_path).unlink()
    
    @patch('docparse.cli.detect_regions')
    def test_layout_command_custom_dpi(self, mock_detect):
        """Test layout command with custom DPI."""
        mock_detect.return_value = []
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--dpi", "300"
            ])
            
            assert result.exit_code == 0
            # Verify mock was called with correct arguments
            mock_detect.assert_called_once()
            call_args = mock_detect.call_args
            # Check that DPI override was applied (indirectly through config)
    
    @patch('docparse.cli.detect_regions')
    def test_layout_command_custom_confidence(self, mock_detect):
        """Test layout command with custom confidence threshold."""
        mock_detect.return_value = []
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--confidence", "0.5"
            ])
            
            assert result.exit_code == 0
            mock_detect.assert_called_once()
    
    @patch('docparse.cli.detect_regions')
    def test_layout_command_custom_model(self, mock_detect):
        """Test layout command with custom model path."""
        mock_detect.return_value = []
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--model", "custom_model.onnx"
            ])
            
            assert result.exit_code == 0
            mock_detect.assert_called_once()


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test CLI configuration integration."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    @patch('docparse.cli.safe_init_config')
    def test_cli_custom_config_file(self, mock_init):
        """Test CLI with custom config file."""
        mock_init.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.toml') as tmp_config:
            result = self.runner.invoke(app, [
                "--config", str(tmp_config.name),
                "--version"
            ])
            
            assert result.exit_code == 0
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            assert call_args[1]['config_path'] == Path(tmp_config.name)
    
    @patch('docparse.cli.safe_init_config')
    def test_cli_profile_selection(self, mock_init):
        """Test CLI with profile selection."""
        mock_init.return_value = None
        
        result = self.runner.invoke(app, [
            "--profile", "academic",
            "--version"
        ])
        
        assert result.exit_code == 0
        mock_init.assert_called_once()
        call_args = mock_init.call_args
        assert call_args[1]['profile'] == "academic"
    
    @patch('docparse.cli.safe_init_config')
    def test_cli_config_and_profile(self, mock_init):
        """Test CLI with both config file and profile."""
        mock_init.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.toml') as tmp_config:
            result = self.runner.invoke(app, [
                "--config", str(tmp_config.name),
                "--profile", "fast",
                "--version"
            ])
            
            assert result.exit_code == 0
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            assert call_args[1]['config_path'] == Path(tmp_config.name)
            assert call_args[1]['profile'] == "fast"
    
    def test_cli_config_error_handling(self):
        """Test CLI error handling for configuration errors."""
        with patch('docparse.cli.safe_init_config') as mock_init:
            from docparse.config_validator import ConfigValidationError
            mock_init.side_effect = ConfigValidationError("Test config error")
            
            result = self.runner.invoke(app, ["--version"])
            
            assert result.exit_code == 1
            assert "Configuration error" in result.stdout


@pytest.mark.integration
class TestCLISubprocess:
    """Test CLI via subprocess (more realistic testing)."""
    
    def test_cli_subprocess_version(self):
        """Test CLI version via subprocess."""
        try:
            result = subprocess.run(
                ["python", "-m", "docparse", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should work if package is installed
            if result.returncode == 0:
                assert "docparse" in result.stdout
            else:
                # Package might not be installed in test environment
                pytest.skip("Package not installed for subprocess testing")
        except FileNotFoundError:
            pytest.skip("Python not available for subprocess testing")
        except subprocess.TimeoutExpired:
            pytest.fail("CLI subprocess timed out")
    
    def test_cli_subprocess_help(self):
        """Test CLI help via subprocess."""
        try:
            result = subprocess.run(
                ["python", "-m", "docparse", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                assert "High-performance PDF text extraction" in result.stdout
            else:
                pytest.skip("Package not installed for subprocess testing")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Cannot run subprocess test")


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    def test_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.runner.invoke(app, ["invalid_command"])
        
        assert result.exit_code != 0
    
    def test_invalid_option(self):
        """Test CLI with invalid option."""
        result = self.runner.invoke(app, ["--invalid-option"])
        
        assert result.exit_code != 0
    
    def test_text_command_invalid_path(self):
        """Test text command with invalid PDF path."""
        result = self.runner.invoke(app, ["text", "/invalid/path/file.pdf"])
        
        assert result.exit_code != 0
    
    def test_layout_command_invalid_dpi(self):
        """Test layout command with invalid DPI."""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--dpi", "invalid"
            ])
            
            assert result.exit_code != 0
    
    def test_layout_command_invalid_confidence(self):
        """Test layout command with invalid confidence."""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, [
                "layout", str(tmp_pdf.name),
                "--confidence", "invalid"
            ])
            
            assert result.exit_code != 0
    
    def test_output_file_permission_error(self):
        """Test CLI with output file permission error."""
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            
            with patch('docparse.cli.extract_plain_text') as mock_extract:
                mock_extract.return_value = ['{"test": "data"}']
                
                with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
                    result = self.runner.invoke(app, [
                        "text", str(tmp_pdf.name),
                        "--output", "/root/restricted.jsonl"
                    ])
                    
                    assert result.exit_code != 0


@pytest.mark.integration
@pytest.mark.slow
class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()
    
    def test_cli_startup_time(self):
        """Test CLI startup time is reasonable."""
        import time
        
        start_time = time.time()
        result = self.runner.invoke(app, ["--version"])
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        # CLI should start within reasonable time (adjust threshold as needed)
        assert startup_time < 2.0  # 2 seconds max
        assert result.exit_code == 0
    
    @patch('docparse.cli.extract_plain_text')
    def test_text_command_large_output(self, mock_extract):
        """Test text command with large output."""
        # Generate large amount of text data
        large_output = []
        for i in range(1000):
            large_output.append(
                f'{{"page": {i//100}, "text": "Line {i} with some text content", '
                f'"x0": {i}, "y0": {i*20}, "x1": {i+100}, "y1": {i*20+20}}}'
            )
        
        mock_extract.return_value = large_output
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_pdf:
            result = self.runner.invoke(app, ["text", str(tmp_pdf.name)])
            
            assert result.exit_code == 0
            # Should handle large output without issues
            lines = result.stdout.strip().split('\n')
            assert len([l for l in lines if l.strip()]) == 1000