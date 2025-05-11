import numpy as np
from PIL import Image
from pdf_backend_pdfium import render_page

w, h, raw = render_page("sample.pdf", 0, 224)
arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
Image.fromarray(arr, "RGB").save("page0.png")
print("Wrote page0.png")
