// src/lib.rs

use once_cell::sync::OnceCell;
use pdfium_render::prelude::{Pdfium, PdfiumError as PdfiumLibError};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use serde::Serialize;
use std::path::Path;

// Singleton to hold our Pdfium instance
static PDFIUM: OnceCell<Pdfium> = OnceCell::new();

fn get_pdfium() -> PyResult<&'static Pdfium> {
    PDFIUM.get_or_try_init(|| {
        let bindings = Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path("./")
        )
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e: PdfiumLibError| PyRuntimeError::new_err(
            format!("Failed to bind to Pdfium library: {}", e)
        ))?;
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
        .map_err(|e| PyRuntimeError::new_err(
            format!("Failed to load PDF '{}': {}", path, e)
        ))?;

    let mut all_lines_json = Vec::new();

    for (page_idx, page) in document.pages().iter().enumerate() {
        let text_page = page
            .text()
            .map_err(|e| PyRuntimeError::new_err(
                format!("Failed to get text for page {}: {}", page_idx, e)
            ))?;

        let mut chars_on_page = Vec::new();
        for pdf_char in text_page.chars().iter() {
            let ch = pdf_char.unicode_char().unwrap_or('\u{FFFD}');
            let rect = pdf_char.loose_bounds()
                .map_err(|e| PyRuntimeError::new_err(
                    format!("Failed to get char bounds on page {}: {}", page_idx, e)
                ))?;
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

        chars_on_page.sort_by(|a, b| {
            b.y.partial_cmp(&a.y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        });
        
        let mut page_lines_buffer: Vec<String> = Vec::new();
        let mut processed_char_indices = vec![false; chars_on_page.len()];

        for i in 0..chars_on_page.len() {
            if processed_char_indices[i] {
                continue;
            }

            let anchor_char = &chars_on_page[i];
            let mut current_line_chars = vec![anchor_char.clone()];
            processed_char_indices[i] = true;

            let anchor_y_center = anchor_char.y + anchor_char.height / 2.0;
            let anchor_height = anchor_char.height;

            for j in (i + 1)..chars_on_page.len() {
                if processed_char_indices[j] {
                    continue;
                }

                let candidate_char = &chars_on_page[j];
                let candidate_y_center = candidate_char.y + candidate_char.height / 2.0;
                let y_center_diff = (candidate_y_center - anchor_y_center).abs();
                let vertical_tolerance = (anchor_height + candidate_char.height) / 2.0 * 0.5;

                if y_center_diff < vertical_tolerance {
                    current_line_chars.push(candidate_char.clone());
                    processed_char_indices[j] = true;
                }
            }

            current_line_chars.sort_by(|a, b| {
                a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut line_text = String::new();
            let mut prev_char_end_x: Option<f32> = None;
            for ci in &current_line_chars {
                if let Some(px_val) = prev_char_end_x {
                    let gap = ci.x - px_val;
                    let space_threshold_height_based = ci.height * 0.20;
                    let space_threshold_width_based = ci.width * 0.40;
                    let min_sensible_gap: f32 = 1.0;
                    if gap > min_sensible_gap
                        .max(space_threshold_height_based.max(space_threshold_width_based)) {
                        line_text.push(' ');
                    }
                }
                line_text.push(ci.ch);
                prev_char_end_x = Some(ci.x + ci.width);
            }

            if line_text.trim().is_empty() {
                continue;
            }

            let x0 = current_line_chars.iter().map(|c| c.x).fold(f32::INFINITY, f32::min);
            let y0 = current_line_chars.iter().map(|c| c.y).fold(f32::INFINITY, f32::min);
            let x1 = current_line_chars.iter().map(|c| c.x + c.width).fold(f32::NEG_INFINITY, f32::max);
            let y1 = current_line_chars.iter().map(|c| c.y + c.height).fold(f32::NEG_INFINITY, f32::max);

            let line_obj = Line { page: page_idx + 1, text: line_text, x0, y0, x1, y1 };
            let line_json = serde_json::to_string(&line_obj)
                .map_err(|e| PyRuntimeError::new_err(
                    format!("Failed to serialize line to JSON: {}", e)
                ))?;
            page_lines_buffer.push(line_json);
        }
        all_lines_json.extend(page_lines_buffer);
    }

    Ok(all_lines_json)
}

#[pymodule]
fn pdf_backend_pdfium(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_plain_text, m)?)?;
    Ok(())
}
