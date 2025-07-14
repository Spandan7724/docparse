# src/docparse/cli.py

import sys
import json
from pathlib import Path
from typing import Optional

import typer
from docparse import __version__
from pdf_backend_pdfium import (
    extract_plain_text,
    render_page,
    page_count,
    init_config_with_profile,
)
from .layout import raster_page, detect_regions
from .config import (
    init_config,
    get_dpi,
    get_model_path,
    get_confidence_threshold,
)
from .config_validator import safe_init_config, ConfigValidationError

app = typer.Typer(add_completion=False)

@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to TOML configuration file (default: config.toml)"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Configuration profile: academic, fast, forms"),
):
    """
    High-performance PDF text extraction and layout analysis.
    
    A hybrid Python/Rust tool that combines fast PDF processing with ML-powered layout detection and intelligent text extraction capabilities.
    
    Configuration priority: CLI arguments > environment variables > config file > defaults.
    
    Available profiles: academic (high DPI, strict thresholds), fast (low DPI, CPU-only), forms (optimized for forms).
    
    Examples: 'docparse --profile academic layout doc.pdf', 'docparse text paper.pdf -o results.jsonl'
    """
    if version:
        typer.echo(f"docparse {__version__}")
        raise typer.Exit()
    
    # Initialize configuration with validation
    try:
        safe_init_config(config_path=config, profile=profile, strict=False)
        # Also initialize Rust config
        init_config_with_profile(profile)
    except ConfigValidationError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error initializing configuration: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def text(
    path: Path,
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Write JSON lines to this file; defaults to stdout"
    ),
):
    """
    Extract text with intelligent clustering and space detection.
    
    Uses advanced character clustering algorithms to extract text from PDFs with proper spacing and line detection. Outputs one JSON object per line containing text content, character-level bounding boxes, and line clustering data.
    
    Configure via config.toml [text_extraction] section, environment variables (DOCPARSE_TEXT_EXTRACTION__*), or CLI profiles.
    """
    if extract_plain_text is None:
        typer.echo("Rust backend not built; run `maturin develop` first.", err=True)
        raise typer.Exit(1)

    writer = open(output, "w", encoding="utf-8") if output else sys.stdout
    for json_line in extract_plain_text(str(path)):
        print(json_line, file=writer)
    if output:
        writer.close()

@app.command()
def layout(
    path: Path,
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="JSON output file (defaults to stdout)"
    ),
    dpi: Optional[int] = typer.Option(
        None, "-d", "--dpi", help="Render DPI (overrides config default)"
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m", help="ONNX layout model path (overrides config default)"
    ),
    confidence: Optional[float] = typer.Option(
        None, "--confidence", help="Confidence threshold 0.0-1.0 (overrides config default)"
    ),
):
    """
    Detect document layout regions using YOLOv8 object detection.
    
    Analyzes PDF pages to identify and classify document regions (text blocks, tables, figures, titles, lists) using a pre-trained YOLOv8 model. Outputs JSON Lines with bounding box coordinates, region types, and confidence scores.
    
    Configure via CLI arguments, environment variables (DOCPARSE_LAYOUT_DETECTION__*), config.toml [layout_detection] section, or profiles.
    """
    # Get configuration values with CLI overrides
    actual_dpi = get_dpi(dpi)
    actual_model_path = get_model_path(model)
    actual_confidence = get_confidence_threshold(confidence)
    
    # Validate model path exists
    if not actual_model_path.exists():
        typer.echo(f"Model file not found: {actual_model_path}", err=True)
        raise typer.Exit(1)
    
    writer = open(output, "w", encoding="utf-8") if output else sys.stdout

    # get total pages from Rust
    total = page_count(str(path))

    for i in range(total):
        # rasterise the page (returns H×W×3 numpy array)
        img = raster_page(str(path), i, actual_dpi)
        # detect regions with your ONNX model
        regs = detect_regions(img, str(actual_model_path), confidence_threshold=actual_confidence)
        print(json.dumps({"page": i+1, "regions": regs}), file=writer)

    if output:
        writer.close()
