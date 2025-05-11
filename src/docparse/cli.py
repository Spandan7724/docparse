# src/docparse/cli.py

import sys
import json
from pathlib import Path

import typer
from docparse import __version__
from pdf_backend_pdfium import (
    extract_plain_text,
    render_page,
    page_count,
)
from .layout import raster_page, detect_regions

app = typer.Typer(add_completion=False)

@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    if version:
        typer.echo(f"docparse {__version__}")
        raise typer.Exit()

@app.command()
def text(
    path: Path,
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Write JSON lines to this file; defaults to stdout"
    ),
):
    """
    Extract text with glyph clustering → one JSON object per line.
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
    output: Path = typer.Option(
        None, "--output", "-o", help="JSON output file"
    ),
    dpi: int = typer.Option(
        224,"-d","--dpi",help="Render DPI (defaults to 224)",show_default=True,
    ),
    model: Path = typer.Option(
        Path("models/yolov8s-doclaynet.onnx"),
        "--model", "-m",
        help="ONNX layout model (defaults to models/yolov8s-doclaynet.onnx)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """
    Detect layout regions (tables, figures, etc.) on each page.
    """
    writer = open(output, "w", encoding="utf-8") if output else sys.stdout

    # get total pages from Rust
    total = page_count(str(path))

    for i in range(total):
        # rasterise the page (returns H×W×3 numpy array)
        img = raster_page(str(path), i, dpi)
        # detect regions with your ONNX model
        regs = detect_regions(img, str(model))
        print(json.dumps({"page": i+1, "regions": regs}), file=writer)

    if output:
        writer.close()
