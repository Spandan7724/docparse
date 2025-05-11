// rust/pdf_backend_pdfium/src/rasteriser.rs

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pdfium_render::prelude::*;
use std::path::Path;

/// Renders a single PDF page to raw RGB bytes at the given DPI.
/// Returns (width_px, height_px, raw_rgb_bytes).
#[pyfunction]
pub fn render_page<'py>(
    py: Python<'py>,
    path: &str,
    page_index: usize,
    dpi: u32,
) -> PyResult<(usize, usize, &'py PyBytes)> {
    // 1. Get the Pdfium singleton
    let pdfium = super::get_pdfium()?;

    // 2. Load the document
    let doc = pdfium
        .load_pdf_from_file(Path::new(path), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open PDF '{}': {}", path, e)))?;

    let pages = doc.pages();
    let page_count = pages.len() as usize;
    if page_index >= page_count {
        return Err(PyRuntimeError::new_err(format!(
            "Page index {} out of range (0..{})",
            page_index, page_count
        )));
    }

    // Convert usize -> PdfPageIndex (u16)
    let page_idx_u16: PdfPageIndex = (page_index as u16)
        .try_into()
        .expect("Page index fits in u16");
    let page = pages
        .get(page_idx_u16)
        .map_err(|e| PyRuntimeError::new_err(format!("Unable to get page {}: {}", page_index, e)))?;

    // 3. Compute target pixel dimensions from page size in PDF points (1pt = 1/72in)
    let width_pts = page.width().value;
    let height_pts = page.height().value;
    let width_px = ((width_pts * dpi as f32) / 72.0).round() as u32;
    let height_px = ((height_pts * dpi as f32) / 72.0).round() as u32;

    // Convert u32 -> i32 for PdfRenderConfig API
    let width: i32 = width_px
        .try_into()
        .map_err(|_| PyRuntimeError::new_err(format!("Width {} exceeds i32 range", width_px)))?;
    let height: i32 = height_px
        .try_into()
        .map_err(|_| PyRuntimeError::new_err(format!("Height {} exceeds i32 range", height_px)))?;

    // 4. Configure rendering
    let config = PdfRenderConfig::new()
        .set_target_width(width)
        .set_maximum_height(height)
        .rotate_if_landscape(PdfPageRenderRotation::None, false);

    // 5. Render the page
    let bmp = page
        .render_with_config(&config)
        .map_err(|e| PyRuntimeError::new_err(format!("Render failed: {}", e)))?;

    // 6. Extract raw RGB bytes
    let img = bmp.as_image().into_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw: Vec<u8> = img.into_raw();

    // 7. Wrap raw bytes in Python bytes object
    let py_bytes = PyBytes::new(py, &raw);

    Ok((w, h, py_bytes))
}

/// Return the number of pages in the given PDF.
#[pyfunction]
pub fn page_count(path: &str) -> PyResult<usize> {
    // 1. Bind to Pdfium
    let pdfium = super::get_pdfium()?;

    // 2. Load the document
    let doc = pdfium
        .load_pdf_from_file(Path::new(path), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open PDF '{}': {}", path, e)))?;

    // 3. pages().len() is a u16 under the hood; cast to usize
    Ok(doc.pages().len() as usize)
}