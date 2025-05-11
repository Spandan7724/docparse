from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("docparse")
except PackageNotFoundError:
    __version__ = "0.0.0"

try:
    from pdf_backend_pdfium import extract_plain_text  # noqa: F401
except ModuleNotFoundError:
    # Rust backend not yet built
    pass

from .text import extract as extract_text