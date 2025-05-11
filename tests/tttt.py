from pathlib import Path
from docparse.text import extract

# for line in extract(Path("sample.pdf")):
#     print(f"[Page {line.page}] {line.text}")


lines = list(extract(Path("sample.pdf")))
for line in lines:
    print(line)