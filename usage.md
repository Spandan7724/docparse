# Docparse Usage Guide

Docparse is a fast PDF text extraction and layout analysis tool with a hybrid Python/Rust architecture. This guide covers all CLI commands, configuration options, and usage examples.

## Installation

```bash
# Install with CPU support
pip install -e ".[cpu]"

# Install with GPU support (recommended)
pip install -e ".[gpu]"

# Development build
maturin develop
```

## Basic Commands

### Text Extraction

Extract text with precise character positioning from PDF documents:

```bash
# Extract text to stdout
python -m docparse text document.pdf

# Save text to file
python -m docparse text document.pdf --output text_output.jsonl

# Extract from large document
python -m docparse text large_report.pdf -o report_text.jsonl
```

**Output format**: JSON Lines with text and bounding boxes
```json
{"page": 1, "text": "Introduction", "x0": 72.0, "y0": 720.0, "x1": 150.0, "y1": 735.0}
{"page": 1, "text": "This document presents...", "x0": 72.0, "y0": 700.0, "x1": 400.0, "y1": 715.0}
```

### Layout Detection

Detect document regions (tables, figures, text blocks) using deep learning:

```bash
# Detect layout regions
python -m docparse layout document.pdf

# Save layout analysis
python -m docparse layout document.pdf --output layout_results.json

# High-quality analysis
python -m docparse layout document.pdf --dpi 300 --confidence 0.4

# Fast analysis (lower quality)
python -m docparse layout document.pdf --dpi 150 --confidence 0.2
```

**Output format**: JSON with detected regions per page
```json
{"page": 1, "regions": [
  {"bbox": [72.0, 600.0, 200.0, 50.0], "score": 0.95, "label": 1},
  {"bbox": [300.0, 580.0, 150.0, 80.0], "score": 0.87, "label": 2}
]}
```

## Global Options

Available for all commands:

```bash
# Show version
python -m docparse --version

# Use custom configuration file
python -m docparse --config my-config.toml layout document.pdf

# Use configuration profile
python -m docparse --profile academic layout document.pdf

# Combine options
python -m docparse --config custom.toml --profile fast text document.pdf
```

## Command-Specific Options

### Text Extraction Options

```bash
# Basic text extraction
python -m docparse text [OPTIONS] PDF_FILE

# Options:
#   --output, -o PATH    Output file (default: stdout)
```

### Layout Detection Options

```bash
# Layout analysis with options
python -m docparse layout [OPTIONS] PDF_FILE

# Options:
#   --output, -o PATH           Output file (default: stdout)
#   --dpi, -d INTEGER          Render DPI (overrides config)
#   --model, -m PATH           ONNX model path (overrides config)
#   --confidence FLOAT         Confidence threshold (overrides config)
```

## Configuration Profiles

### Academic Profile (High Quality)
Optimized for academic papers and research documents:

```bash
python -m docparse --profile academic layout paper.pdf
```

**Settings:**
- DPI: 300 (high resolution)
- Confidence: 0.4 (stricter detection)
- Line clustering: 0.20 (precise text grouping)

### Fast Profile (Quick Processing)
Optimized for speed over quality:

```bash
python -m docparse --profile fast layout document.pdf
```

**Settings:**
- DPI: 150 (lower resolution)
- Input size: 512 (smaller model input)
- CPU-only execution

### Forms Profile (Structured Documents)
Optimized for forms and structured documents:

```bash
python -m docparse --profile forms layout form.pdf
```

**Settings:**
- Confidence: 0.5 (very strict)
- IoU threshold: 0.3 (less overlap tolerance)
- DPI: 150 (adequate for forms)

## Environment Variables

Override configuration using environment variables:

```bash
# Layout detection settings
export DOCPARSE_LAYOUT_DETECTION__DPI=300
export DOCPARSE_LAYOUT_DETECTION__CONFIDENCE_THRESHOLD=0.4
export DOCPARSE_LAYOUT_DETECTION__MODEL_PATH="/path/to/custom/model.onnx"

# Text extraction settings
export DOCPARSE_TEXT_EXTRACTION__LINE_CLUSTERING_TOLERANCE=0.3
export DOCPARSE_TEXT_EXTRACTION__SPACE_THRESHOLD_HEIGHT=0.25

# Runtime settings
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CPUExecutionProvider"]'

# Run with environment overrides
python -m docparse layout document.pdf
```

## Advanced Usage Examples

### Batch Processing
Process multiple documents:

```bash
# Process all PDFs in directory
for pdf in *.pdf; do
    echo "Processing $pdf..."
    python -m docparse text "$pdf" --output "text_${pdf%.pdf}.jsonl"
    python -m docparse layout "$pdf" --output "layout_${pdf%.pdf}.json"
done
```

### High-Quality Analysis Pipeline
For critical document analysis:

```bash
# Step 1: High-quality text extraction
python -m docparse --profile academic text document.pdf -o document_text.jsonl

# Step 2: High-resolution layout analysis
python -m docparse --profile academic layout document.pdf \
    --dpi 300 --confidence 0.45 -o document_layout.json

# Step 3: Custom model analysis
python -m docparse layout document.pdf \
    --model custom_model.onnx --dpi 400 --confidence 0.5
```

### GPU Acceleration
Force GPU usage for faster processing:

```bash
# Set GPU-only execution
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CUDAExecutionProvider"]'

# Process large document with GPU
python -m docparse layout large_document.pdf --dpi 300
```

### Memory-Constrained Processing
For large documents or limited memory:

```bash
# Use lower resolution and CPU-only
export DOCPARSE_LAYOUT_DETECTION__DPI=150
export DOCPARSE_LAYOUT_DETECTION__INPUT_SIZE=512
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CPUExecutionProvider"]'

python -m docparse layout huge_document.pdf
```

## Configuration Examples

### Custom Configuration File

Create `my-config.toml`:

```toml
[text_extraction]
line_clustering_tolerance = 0.3
space_threshold_height = 0.25
space_threshold_width = 0.35

[layout_detection]
dpi = 250
confidence_threshold = 0.35
iou_threshold = 0.4
model_path = "models/custom-model.onnx"

[runtime]
execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
thread_count = 4

[paths]
models_directory = "/opt/docparse/models/"
cache_directory = "/tmp/docparse_cache/"
```

Use custom configuration:

```bash
python -m docparse --config my-config.toml layout document.pdf
```

### Performance Tuning

For maximum performance:

```bash
# GPU acceleration with high DPI
export DOCPARSE_LAYOUT_DETECTION__DPI=300
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CUDAExecutionProvider"]'
export DOCPARSE_RUNTIME__THREAD_COUNT=8

python -m docparse layout document.pdf
```

For CPU-optimized processing:

```bash
# CPU-only with parallel threads
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CPUExecutionProvider"]'
export DOCPARSE_RUNTIME__THREAD_COUNT=4
export DOCPARSE_LAYOUT_DETECTION__DPI=200

python -m docparse layout document.pdf
```

## Output Processing

### Working with Text Output

```python
import json
from pathlib import Path

# Read extracted text
with open('document_text.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        print(f"Page {data['page']}: {data['text']}")
        print(f"  Position: ({data['x0']:.1f}, {data['y0']:.1f}) to ({data['x1']:.1f}, {data['y1']:.1f})")
```

### Working with Layout Output

```python
import json

# Read layout analysis
with open('document_layout.json', 'r') as f:
    for line in f:
        page_data = json.loads(line)
        print(f"Page {page_data['page']} has {len(page_data['regions'])} regions:")
        
        for region in page_data['regions']:
            bbox = region['bbox']
            print(f"  Region {region['label']}: confidence={region['score']:.2f}")
            print(f"    Location: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")
```

## Troubleshooting

### Common Issues

**Model not found:**
```bash
# Check if model exists
ls -la models/yolov8s-doclaynet.onnx

# Use custom model path
python -m docparse layout document.pdf --model /path/to/model.onnx
```

**CUDA not available:**
```bash
# Force CPU execution
export DOCPARSE_RUNTIME__EXECUTION_PROVIDERS='["CPUExecutionProvider"]'
python -m docparse layout document.pdf
```

**Memory issues:**
```bash
# Reduce memory usage
python -m docparse layout document.pdf --dpi 150
```

**Configuration errors:**
```bash
# Validate configuration
python -c "from docparse.config_validator import validate_environment; print(validate_environment())"
```

### Performance Optimization

**For speed:**
- Use `--profile fast`
- Lower DPI (150-200)
- CPU-only execution for small documents
- Reduce confidence threshold

**For quality:**
- Use `--profile academic` 
- Higher DPI (300-400)
- GPU execution
- Higher confidence threshold (0.4-0.5)

**For memory efficiency:**
- Lower input size (512)
- CPU execution
- Process pages individually

## Integration Examples

### Shell Script Integration

```bash
#!/bin/bash
# document_processor.sh

PDF_FILE="$1"
OUTPUT_DIR="output"

mkdir -p "$OUTPUT_DIR"

echo "Processing $PDF_FILE..."

# Extract text
python -m docparse text "$PDF_FILE" \
    --output "$OUTPUT_DIR/$(basename "$PDF_FILE" .pdf)_text.jsonl"

# Analyze layout
python -m docparse --profile academic layout "$PDF_FILE" \
    --output "$OUTPUT_DIR/$(basename "$PDF_FILE" .pdf)_layout.json"

echo "Results saved to $OUTPUT_DIR"
```

### Python Script Integration

```python
#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path

def process_document(pdf_path, output_dir):
    """Process a PDF document with docparse."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    base_name = pdf_path.stem
    
    # Extract text
    text_output = output_dir / f"{base_name}_text.jsonl"
    subprocess.run([
        "python", "-m", "docparse", "text", str(pdf_path),
        "--output", str(text_output)
    ], check=True)
    
    # Analyze layout
    layout_output = output_dir / f"{base_name}_layout.json"
    subprocess.run([
        "python", "-m", "docparse", "--profile", "academic", 
        "layout", str(pdf_path), "--output", str(layout_output)
    ], check=True)
    
    return text_output, layout_output

# Usage
text_file, layout_file = process_document("document.pdf", "output")
print(f"Text extracted to: {text_file}")
print(f"Layout analyzed to: {layout_file}")
```

This usage guide covers all the main functionality and provides practical examples for getting started with docparse. For more advanced configuration options, see the `config.toml` file and the configuration system documentation in `CLAUDE.md`.