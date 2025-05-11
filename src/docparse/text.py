# src/docparse/text.py

import json
from typing import Iterator, NamedTuple
from pathlib import Path
from pdf_backend_pdfium import extract_plain_text

class PageText(NamedTuple):
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

def extract(path: Path) -> Iterator[PageText]:
    """
    Yield a PageText namedtuple for every line in `path`.
    """
    for line_json in extract_plain_text(str(path)):
        data = json.loads(line_json)
        yield PageText(
            page = data["page"],
            text = data["text"],
            x0 = data["x0"],
            y0 = data["y0"],
            x1 = data["x1"],
            y1 = data["y1"],
        )
