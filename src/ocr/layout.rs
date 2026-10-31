//! Indra OrderNet features, pairwise reading order, and layout output.

use anyhow::Result;
use ort::{inputs, session::Session, value::Tensor};
use serde::Serialize;
use unicode_width::UnicodeWidthChar;

use super::Line;

const LABELS: [&str; 10] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];

#[derive(Debug, Clone, Serialize)]
pub struct Block {
    pub label: &'static str,
    pub bbox: [f32; 4],
    pub text: String,
}

pub struct Layout {
    pub lines: Vec<Line>,
    pub blocks: Vec<Block>,
    pub markdown: String,
}

pub fn assemble(
    session: Option<&mut Session>,
    lines: Vec<Line>,
    page_width: u32,
    page_height: u32,
) -> Result<Layout> {
    if lines.is_empty() {
        return Ok(Layout {
            lines,
            blocks: Vec::new(),
            markdown: String::new(),
        });
    }
    let n = lines.len();
    let (order, classes, same) = if let Some(session) = session.filter(|_| n > 1 && n <= 1000) {
        let features = line_features(&lines, page_width as f32, page_height as f32);
        let input = Tensor::from_array(([1, n, 16], features))?;
        let mask = Tensor::from_array(([1, n], vec![true; n]))?;
        let outputs = session.run(inputs!["feats" => input, "mask" => mask])?;
        let (_, logits) = outputs["a"].try_extract_tensor::<f32>()?;
        let (_, class_logits) = outputs["cls"].try_extract_tensor::<f32>()?;
        let (_, same_logits) = outputs["same"].try_extract_tensor::<f32>()?;
        let order = decode_order(logits, n);
        let classes = (0..n)
            .map(|index| {
                (0..LABELS.len())
                    .max_by(|&a, &b| {
                        class_logits[index * LABELS.len() + a]
                            .total_cmp(&class_logits[index * LABELS.len() + b])
                    })
                    .unwrap_or(0)
            })
            .collect();
        (order, classes, same_logits.to_vec())
    } else {
        ((0..n).collect(), vec![0; n], vec![0.0; n * n])
    };
    Ok(assemble_ordered(lines, order, classes, same))
}

fn assemble_ordered(
    lines: Vec<Line>,
    order: Vec<usize>,
    classes: Vec<usize>,
    same: Vec<f32>,
) -> Layout {
    let n = lines.len();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &index in &order {
        if groups
            .last()
            .is_some_and(|group| same[group[group.len() - 1] * n + index] > 0.0)
        {
            groups.last_mut().expect("checked above").push(index);
        } else {
            groups.push(vec![index]);
        }
    }
    let blocks = groups
        .iter()
        .map(|group| block(group, &lines, &classes))
        .collect::<Vec<_>>();
    let markdown = markdown(&blocks, false);
    let ordered_lines = order
        .into_iter()
        .map(|index| lines[index].clone())
        .collect();
    Layout {
        lines: ordered_lines,
        blocks,
        markdown,
    }
}

fn line_features(lines: &[Line], width: f32, height: f32) -> Vec<f32> {
    let mut heights = lines
        .iter()
        .map(|line| (line.bbox[3] - line.bbox[1]).max(0.001))
        .collect::<Vec<_>>();
    heights.sort_by(f32::total_cmp);
    let median = heights[heights.len() / 2];
    let mut features = Vec::with_capacity(lines.len() * 16);
    for line in lines {
        let [x0, y0, x1, y1] = line.bbox;
        let w = (x1 - x0).max(0.001);
        let h = (y1 - y0).max(0.001);
        features.extend_from_slice(&[
            x0 / width,
            y0 / height,
            x1 / width,
            y1 / height,
            (x0 + x1) / (2.0 * width),
            (y0 + y1) / (2.0 * height),
            w / width,
            (h / median).ln(),
            (w / h).ln() / 4.0,
            median / height * 20.0,
        ]);
        let text = line.text.trim();
        let len = text.chars().count().max(1);
        let display_len: usize = text
            .chars()
            .map(|c| UnicodeWidthChar::width(c).unwrap_or(1))
            .sum();
        let first = text.chars().next();
        let last = text.chars().last();
        features.extend_from_slice(&[
            (display_len as f32).ln_1p() / 5.0,
            text.chars().filter(|c| c.is_numeric()).count() as f32 / len as f32,
            f32::from(first.is_some_and(|c| "•-–▪◦*·→".contains(c))),
            f32::from(text.ends_with('-')),
            f32::from(last.is_some_and(|c| ".!?:;。！？：；．".contains(c))),
            f32::from(first.is_some_and(char::is_uppercase)),
        ]);
    }
    features
}

fn decode_order(logits: &[f32], n: usize) -> Vec<usize> {
    let probability = logits
        .iter()
        .map(|&value| -(-value).exp().ln_1p())
        .collect::<Vec<_>>();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        let score = |index: usize| {
            (0..n)
                .map(|other| probability[other * n + index] - probability[index * n + other])
                .sum::<f32>()
        };
        score(left).total_cmp(&score(right))
    });
    for _ in 0..4 {
        let mut improved = false;
        for line in order.clone() {
            let current = order
                .iter()
                .position(|&index| index == line)
                .expect("line in order");
            let rest = order
                .iter()
                .copied()
                .filter(|&index| index != line)
                .collect::<Vec<_>>();
            let gain = |position: usize| {
                rest.iter()
                    .enumerate()
                    .map(|(index, &other)| {
                        if index < position {
                            probability[other * n + line]
                        } else {
                            probability[line * n + other]
                        }
                    })
                    .sum::<f32>()
            };
            let mut running_gain = rest
                .iter()
                .map(|&other| f64::from(probability[line * n + other]))
                .sum::<f64>();
            let mut gains = Vec::with_capacity(rest.len() + 1);
            gains.push(running_gain);
            let mut magnitude = 0.0;
            for &other in &rest {
                // Moving past one line changes only this pair's contribution.
                let before = f64::from(probability[other * n + line]);
                let after = f64::from(probability[line * n + other]);
                running_gain += before - after;
                gains.push(running_gain);
                magnitude += before.abs().max(after.abs());
            }
            // Bound the original f32 sums' rounding error, including subnormals.
            // Recheck ambiguous candidates with the original summation order so
            // near ties, last-position tie breaking, and the threshold stay exact.
            // Well-separated gains need only one O(n) verification; ambiguous or
            // non-finite inputs retain the original O(n²) worst case.
            let error = 2.0
                * (rest.len() + 1) as f64
                * (f64::from(f32::EPSILON) * magnitude + f64::from(f32::from_bits(1)));
            let largest = gains.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let finite = error.is_finite() && gains.iter().all(|gain| gain.is_finite());
            let best = (0..gains.len())
                .filter(|&position| !finite || gains[position] >= largest - 2.0 * error)
                .map(|position| (position, gain(position)))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(current, |(position, _)| position);
            if gain(best) > gain(current) + 1e-6 {
                order = rest;
                order.insert(best, line);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

fn block(indices: &[usize], lines: &[Line], classes: &[usize]) -> Block {
    let mut counts = [0usize; LABELS.len()];
    let mut bbox = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let mut texts = Vec::new();
    for &index in indices {
        counts[classes[index]] += 1;
        let line = &lines[index];
        bbox[0] = bbox[0].min(line.bbox[0]);
        bbox[1] = bbox[1].min(line.bbox[1]);
        bbox[2] = bbox[2].max(line.bbox[2]);
        bbox[3] = bbox[3].max(line.bbox[3]);
        if !line.text.is_empty() {
            texts.push(line.text.as_str());
        }
    }
    let class = (0..LABELS.len())
        .max_by_key(|&index| counts[index])
        .unwrap_or(0);
    Block {
        label: LABELS[class],
        bbox,
        text: texts.join(" "),
    }
}

pub(super) fn markdown(blocks: &[Block], keep_furniture: bool) -> String {
    blocks
        .iter()
        .filter_map(|block| render_markdown(block, keep_furniture))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_markdown(block: &Block, keep_furniture: bool) -> Option<String> {
    if block.text.is_empty()
        || (!keep_furniture && matches!(block.label, "Page-Header" | "Page-Footer"))
    {
        return None;
    }
    Some(match block.label {
        "Section-Header" => format!("## {}", block.text),
        _ => block.text.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn furniture_affects_markdown_without_removing_structured_blocks() {
        let blocks = vec![
            Block {
                label: "Page-Header",
                bbox: [0.0; 4],
                text: "letterhead".into(),
            },
            Block {
                label: "Text",
                bbox: [0.0; 4],
                text: "body".into(),
            },
            Block {
                label: "Page-Footer",
                bbox: [0.0; 4],
                text: "page 1".into(),
            },
        ];
        assert_eq!(markdown(&blocks, false), "body");
        assert_eq!(markdown(&blocks, true), "letterhead\n\nbody\n\npage 1");
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn assembled_pages_omit_furniture_from_default_markdown_but_keep_structure() {
        let lines = ["letterhead", "body", "page 1"]
            .into_iter()
            .enumerate()
            .map(|(index, text)| Line {
                text: text.into(),
                bbox: [0.0, index as f32 * 20.0, 100.0, index as f32 * 20.0 + 10.0],
                confidence: 1.0,
            })
            .collect();
        // Class IDs are the OrderNet vocabulary, not preconstructed blocks.
        let layout = assemble_ordered(lines, vec![0, 1, 2], vec![2, 0, 3], vec![0.0; 9]);
        assert_eq!(layout.markdown, "body");
        assert_eq!(layout.lines.len(), 3);
        assert_eq!(layout.blocks.len(), 3);
        assert_eq!(layout.blocks[0].label, "Page-Header");
        assert_eq!(layout.blocks[2].label, "Page-Footer");
        assert_eq!(
            markdown(&layout.blocks, true),
            "letterhead\n\nbody\n\npage 1"
        );
    }

    #[test]
    fn pairwise_decoder_follows_order_logits() {
        let logits = [0.0, 3.0, 3.0, -3.0, 0.0, 3.0, -3.0, -3.0, 0.0];
        assert_eq!(decode_order(&logits, 3), vec![0, 1, 2]);
    }

    #[test]
    fn pairwise_decoder_preserves_last_tied_insertion() {
        let logits = [
            3.0, -3.0, 3.0, 3.0, 3.0, 3.0, -3.0, 3.0, -3.0, 0.0, -3.0, -3.0, 3.0, 0.0, 0.0, -3.0,
        ];
        assert_eq!(decode_order(&logits, 4), vec![1, 0, 3, 2]);
    }

    #[test]
    fn pairwise_decoder_preserves_original_rounding_near_threshold() {
        // This matrix diverges under a plain incremental f32 gain scan.
        let mut seed = 972845047165077966_u64;
        let logits = (0..100)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as i32 as f32 / i32::MAX as f32 * 10.0
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decode_order(&logits, 10),
            vec![8, 1, 4, 6, 9, 7, 2, 5, 3, 0]
        );
    }

    #[test]
    fn pairwise_decoder_keeps_order_without_an_improvement() {
        assert!(decode_order(&[], 0).is_empty());
        assert_eq!(decode_order(&[0.0], 1), vec![0]);
        for value in [0.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            assert_eq!(decode_order(&[value; 9], 3), vec![0, 1, 2]);
        }
    }
}
