# src/docparse/cli.py

import sys
import json
from pathlib import Path

import typer
from docparse import __version__
from pdf_backend_pdfium import extract_plain_text

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
