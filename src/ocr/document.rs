//! Model-independent PDF text-layer validation and page assembly.
//!
//! Coordinates supplied by the PDF renderer must already be converted into the
//! top-left pixel coordinate system used for OCR output. These helpers do not
//! claim parity with upstream Indra's PDF quality classifier.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::{Line, Page, layout};

/// PDF text-layer selection policy. Raster images do not have a text layer.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrMode {
    /// Use a usable text layer, otherwise recognize the page image.
    #[default]
    Auto,
    /// Always recognize page pixels.
    Ocr,
    /// Use the PDF text layer without applying auto quality checks.
    Text,
}

/// Validate dimensions and extracted boxes before page assembly.
pub fn validate_text_lines(lines: &[Line], width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("document page dimensions must be positive");
    }
    for line in lines {
        let [x0, y0, x1, y1] = line.bbox;
        if !line.bbox.iter().all(|value| value.is_finite())
            || x0 < 0.0
            || y0 < 0.0
            || x1 <= x0
            || y1 <= y0
            || x1 > width as f32
            || y1 > height as f32
        {
            bail!("PDF text line has invalid page coordinates");
        }
    }
    Ok(())
}

/// Conservative auto-mode check, independent of a particular writing system.
///
/// Empty or very sparse extraction falls back to OCR. Replacement characters
/// and unexpected control characters indicate a damaged character mapping.
pub fn text_layer_usable(lines: &[Line], width: u32, height: u32) -> bool {
    if validate_text_lines(lines, width, height).is_err() {
        return false;
    }
    let mut visible = 0usize;
    let mut damaged = 0usize;
    for character in lines.iter().flat_map(|line| line.text.chars()) {
        if !character.is_whitespace() {
            visible += 1;
        }
        if character == '\u{fffd}'
            || (character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        {
            damaged += 1;
        }
    }
    visible >= 16 && damaged <= visible / 50
}

/// Assemble an extracted text page without creating ONNX sessions.
pub fn page_from_text_lines(
    lines: Vec<Line>,
    page_index: usize,
    width: u32,
    height: u32,
    keep_furniture: bool,
) -> Result<Page> {
    page_from_lines(lines, page_index, width, height, keep_furniture, "text")
}

/// Assemble a page whose text layer and image OCR have already been merged.
///
/// Callers supply the merged reading order and pixel-coordinate transforms;
/// this function preserves that order and does not silently reorder columns.
pub fn page_from_hybrid_lines(
    lines: Vec<Line>,
    page_index: usize,
    width: u32,
    height: u32,
    keep_furniture: bool,
) -> Result<Page> {
    page_from_lines(lines, page_index, width, height, keep_furniture, "hybrid")
}

fn page_from_lines(
    lines: Vec<Line>,
    page_index: usize,
    width: u32,
    height: u32,
    keep_furniture: bool,
    source: &'static str,
) -> Result<Page> {
    validate_text_lines(&lines, width, height)?;
    let assembled = layout::assemble(None, lines, width, height)?;
    let text = assembled
        .lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Page {
        page_index,
        rotation: 0,
        vertical: false,
        text,
        markdown: layout::markdown(&assembled.blocks, keep_furniture),
        source,
        lines: assembled.lines,
        blocks: assembled.blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, y: f32) -> Line {
        Line {
            text: text.into(),
            bbox: [10.0, y, 250.0, y + 20.0],
            confidence: 1.0,
        }
    }

    #[test]
    fn text_quality_accepts_unicode_without_a_latin_bias() {
        assert!(text_layer_usable(
            &[line("대한성공회 서울교구 교구장 이경호 주교 안내문", 0.0)],
            300,
            400
        ));
        assert!(!text_layer_usable(&[], 300, 400));
        assert!(!text_layer_usable(&[line("tiny", 0.0)], 300, 400));
        assert!(!text_layer_usable(
            &[line(
                "broken mapping \u{fffd}\u{fffd}\u{fffd} text layer",
                0.0
            )],
            300,
            400
        ));
    }

    #[test]
    fn text_page_preserves_extracted_reading_order_and_geometry() {
        let page = page_from_text_lines(
            vec![
                line("second column comes first", 90.0),
                line("then this", 20.0),
            ],
            2,
            300,
            400,
            false,
        )
        .expect("valid page");
        assert_eq!(page.page_index, 2);
        assert_eq!(page.source, "text");
        assert_eq!(page.text, "second column comes first\nthen this");
        assert_eq!(page.lines[0].bbox, [10.0, 90.0, 250.0, 110.0]);
        assert_eq!(page.blocks.len(), 2);
    }

    #[test]
    fn invalid_extracted_boxes_are_never_admitted() {
        for coordinate in [f32::NAN, f32::INFINITY, -1.0, 500.0] {
            let mut input = line("valid content", 0.0);
            input.bbox[0] = coordinate;
            assert!(page_from_text_lines(vec![input], 0, 300, 400, false).is_err());
        }
        assert!(page_from_text_lines(vec![], 0, 0, 400, false).is_err());
    }

    #[test]
    fn hybrid_pages_are_marked_and_empty_text_pages_are_representable() {
        let page = page_from_hybrid_lines(vec![line("image text", 0.0)], 0, 300, 400, true)
            .expect("hybrid page");
        assert_eq!(page.source, "hybrid");
        let empty = page_from_text_lines(vec![], 0, 300, 400, false).expect("empty text layer");
        assert!(empty.text.is_empty());
        assert!(empty.blocks.is_empty());
    }
}
