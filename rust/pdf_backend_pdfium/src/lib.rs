// src/lib.rs
mod rasteriser;

use once_cell::sync::OnceCell;
use pdfium_render::prelude::{Pdfium, PdfiumError as PdfiumLibError};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use serde::Serialize;
use std::path::Path;

static PDFIUM: OnceCell<Pdfium> = OnceCell::new();

fn get_pdfium() -> PyResult<&'static Pdfium> {
    PDFIUM.get_or_try_init(|| {
        let bindings = Pdfium::bind_to_library(
                Pdfium::pdfium_platform_library_name_at_path("./")
            )
            .or_else(|_| Pdfium::bind_to_system_library())
            .map_err(|e: PdfiumLibError| {
                PyRuntimeError::new_err(format!("Failed to bind to Pdfium library: {}", e))
            })?;
        Ok(Pdfium::new(bindings))
    })
}

#[derive(Serialize, Clone, Debug)]
struct CharInfo {
    ch: char,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Serialize, Debug)]
struct Line {
    page: usize,
    text: String,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

#[pyfunction]
fn extract_plain_text(path: &str) -> PyResult<Vec<String>> {
    let pdfium = get_pdfium()?;
    let document = pdfium
        .load_pdf_from_file(Path::new(path), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load PDF '{}': {}", path, e)))?;

    let mut all_lines_json = Vec::new();

    for (page_idx, page) in document.pages().iter().enumerate() {
        let text_page = page
            .text()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get text for page {}: {}", page_idx, e)))?;

        let mut chars_on_page = Vec::new();
        for pdf_char in text_page.chars().iter() {
            let ch = pdf_char.unicode_char().unwrap_or('\u{FFFD}');
            let rect = pdf_char.loose_bounds().map_err(|e| {
                PyRuntimeError::new_err(format!("Char bounds error on page {}: {}", page_idx, e))
            })?;

            let l = rect.left().value;
            let b = rect.bottom().value;
            let w = rect.width().value;
            let h = rect.height().value;

            if w <= 0.0 || h <= 0.0 {
                continue;
            }
            chars_on_page.push(CharInfo { ch, x: l, y: b, width: w, height: h });
        }

        if chars_on_page.is_empty() {
            continue;
        }

        // Sort top-to-bottom, then left-to-right
        chars_on_page.sort_by(|a, b| {
            b.y.partial_cmp(&a.y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut processed = vec![false; chars_on_page.len()];
        for i in 0..chars_on_page.len() {
            if processed[i] {
                continue;
            }
            let anchor = &chars_on_page[i];
            let mut line_chars = vec![anchor.clone()];
            processed[i] = true;

            let anchor_center = anchor.y + anchor.height / 2.0;
            let anchor_h = anchor.height;

            for j in (i + 1)..chars_on_page.len() {
                if processed[j] {
                    continue;
                }
                let cand = &chars_on_page[j];
                let cand_center = cand.y + cand.height / 2.0;
                let tol = (anchor_h + cand.height) * 0.25;
                if (cand_center - anchor_center).abs() < tol {
                    line_chars.push(cand.clone());
                    processed[j] = true;
                }
            }

            // Sort the collected chars left-to-right
            line_chars.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

            // Build the line string with spaces where gaps exceed threshold
            let mut text = String::new();
            let mut prev_end: Option<f32> = None;
            for ci in &line_chars {
                if let Some(px) = prev_end {
                    let gap = ci.x - px;
                    let th_h = ci.height * 0.20;
                    let th_w = ci.width * 0.40;
                    if gap > th_h.max(th_w).max(1.0) {
                        text.push(' ');
                    }
                }
                text.push(ci.ch);
                prev_end = Some(ci.x + ci.width);
            }

            if text.trim().is_empty() {
                continue;
            }

            // Compute bounding box of the line
            let x0 = line_chars.iter().map(|c| c.x).fold(f32::INFINITY, f32::min);
            let y0 = line_chars.iter().map(|c| c.y).fold(f32::INFINITY, f32::min);
            let x1 = line_chars.iter().map(|c| c.x + c.width).fold(f32::NEG_INFINITY, f32::max);
            let y1 = line_chars.iter().map(|c| c.y + c.height).fold(f32::NEG_INFINITY, f32::max);

            let line = Line {
                page: page_idx + 1,
                text,
                x0,
                y0,
                x1,
                y1,
            };
            all_lines_json.push(
                serde_json::to_string(&line)
                    .map_err(|e| PyRuntimeError::new_err(format!("JSON serialize error: {}", e)))?
            );
        }
    }

    Ok(all_lines_json)
}

#[pymodule]
fn pdf_backend_pdfium(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_plain_text, m)?)?;
    m.add_function(wrap_pyfunction!(rasteriser::render_page, m)?)?;
    m.add_function(wrap_pyfunction!(rasteriser::page_count, m)?)?;
    Ok(())
}
