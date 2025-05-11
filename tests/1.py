from pdf_backend_pdfium import render_page
w, h, raw = render_page("sample.pdf", 0, 224)
print(type(raw))         # should print: <class 'bytes'>
