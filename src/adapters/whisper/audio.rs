//! Bounded WAV decoding and Whisper's 16 kHz log-Mel frontend.

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) const SAMPLE_RATE: usize = 16_000;
pub(super) const SAMPLES: usize = SAMPLE_RATE * 30;
const FFT_SIZE: usize = 400;
const HOP: usize = 160;
pub(super) const FRAMES: usize = SAMPLES / HOP;
pub(super) const MELS: usize = 80;

pub(super) fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("expected a RIFF/WAVE audio upload");
    }
    let declared = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    if declared.checked_add(8) != Some(bytes.len()) {
        bail!("invalid WAV RIFF length");
    }
    let mut position = 12usize;
    let mut format = None;
    let mut samples = None;
    while position < bytes.len() {
        if bytes.len() - position < 8 {
            bail!("truncated WAV chunk header");
        }
        let length = u32::from_le_bytes(bytes[position + 4..position + 8].try_into()?) as usize;
        let start = position + 8;
        let end = start
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| anyhow::anyhow!("truncated WAV chunk"))?;
        match &bytes[position..position + 4] {
            b"fmt " => {
                if format.is_some() || length < 16 {
                    bail!("invalid/duplicate WAV format chunk");
                }
                format = Some(&bytes[start..end]);
            }
            b"data" => {
                if samples.is_some() {
                    bail!("multiple WAV data chunks are unsupported");
                }
                samples = Some(&bytes[start..end]);
            }
            _ => {}
        }
        position = end
            .checked_add(length & 1)
            .ok_or_else(|| anyhow::anyhow!("invalid WAV chunk length"))?;
        if position > bytes.len() {
            bail!("missing WAV chunk padding");
        }
    }
    let format = format.ok_or_else(|| anyhow::anyhow!("WAV format chunk missing"))?;
    let samples = samples.ok_or_else(|| anyhow::anyhow!("WAV data chunk missing"))?;
    decode_samples(format, samples)
}

fn decode_samples(format: &[u8], data: &[u8]) -> Result<Vec<f32>> {
    let mut encoding = u16::from_le_bytes(format[..2].try_into()?);
    if encoding == 0xfffe {
        if format.len() < 40
            || u16::from_le_bytes(format[16..18].try_into()?) < 22
            || format[26..40] != [0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]
            || u16::from_le_bytes(format[18..20].try_into()?)
                != u16::from_le_bytes(format[14..16].try_into()?)
        {
            bail!("invalid extensible WAV subformat");
        }
        encoding = u16::from_le_bytes(format[24..26].try_into()?);
    }
    let channels = usize::from(u16::from_le_bytes(format[2..4].try_into()?));
    let rate = u32::from_le_bytes(format[4..8].try_into()?);
    let align = usize::from(u16::from_le_bytes(format[12..14].try_into()?));
    let bits = u16::from_le_bytes(format[14..16].try_into()?);
    let sample_bytes = match (encoding, bits) {
        (1, 16) => 2,
        (3, 32) => 4,
        _ => bail!("supported WAV encodings are PCM16 and IEEE float32"),
    };
    if rate != SAMPLE_RATE as u32
        || !(1..=2).contains(&channels)
        || align != channels * sample_bytes
    {
        bail!("WAV must use 16000 Hz, one or two channels, and valid block alignment");
    }
    if data.is_empty() || !data.len().is_multiple_of(align) || data.len() / align > SAMPLES {
        bail!("WAV must contain complete frames and at most 30 seconds of audio");
    }
    data.chunks_exact(align)
        .map(|frame| {
            let mut sum = 0.0;
            for channel in frame.chunks_exact(sample_bytes) {
                let value = if encoding == 1 {
                    f32::from(i16::from_le_bytes(channel.try_into()?)) / 32768.0
                } else {
                    f32::from_le_bytes(channel.try_into()?)
                };
                if !value.is_finite() {
                    bail!("WAV contains nonfinite audio samples");
                }
                sum += value.clamp(-1.0, 1.0);
            }
            Ok(sum / channels as f32)
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

fn fft(input: &[Complex]) -> Vec<Complex> {
    if input.len() == 1 {
        return vec![input[0]];
    }
    let radix = if input.len().is_multiple_of(2) { 2 } else { 5 };
    let inner = input.len() / radix;
    let branches = (0..radix)
        .map(|offset| {
            fft(&input
                .iter()
                .skip(offset)
                .step_by(radix)
                .copied()
                .collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    (0..input.len())
        .map(|k| {
            let mut sum = Complex::default();
            for (branch, values) in branches.iter().enumerate() {
                let angle = -std::f32::consts::TAU * (branch * k) as f32 / input.len() as f32;
                let value = values[k % inner];
                sum.re += value.re * angle.cos() - value.im * angle.sin();
                sum.im += value.re * angle.sin() + value.im * angle.cos();
            }
            sum
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    if hz < 1000.0 {
        hz / (200.0 / 3.0)
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4_f32.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((mel - 15.0) * (6.4_f32.ln() / 27.0)).exp()
    }
}

fn mel_filters() -> Vec<Vec<f32>> {
    let max_mel = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let centers = (0..MELS + 2)
        .map(|index| mel_to_hz(index as f32 * max_mel / (MELS + 1) as f32))
        .collect::<Vec<_>>();
    (0..MELS)
        .map(|mel| {
            (0..=FFT_SIZE / 2)
                .map(|bin| {
                    let frequency = bin as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
                    let rising = (frequency - centers[mel]) / (centers[mel + 1] - centers[mel]);
                    let falling =
                        (centers[mel + 2] - frequency) / (centers[mel + 2] - centers[mel + 1]);
                    rising.min(falling).max(0.0) * 2.0 / (centers[mel + 2] - centers[mel])
                })
                .collect()
        })
        .collect()
}

pub(super) fn log_mel(samples: &[f32], cancel: &AtomicBool) -> Result<Vec<f32>> {
    if samples.is_empty()
        || samples.len() > SAMPLES
        || samples.iter().any(|value| !value.is_finite())
    {
        bail!("audio must contain 1..480000 finite 16 kHz samples");
    }
    let mut padded = vec![0.0; SAMPLES];
    padded[..samples.len()].copy_from_slice(samples);
    let filters = mel_filters();
    let hann = (0..FFT_SIZE)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT_SIZE as f32).cos())
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; MELS * FRAMES];
    for frame in 0..FRAMES {
        if cancel.load(Ordering::Relaxed) {
            bail!("transcription cancelled");
        }
        let window = (0..FFT_SIZE)
            .map(|offset| {
                let position =
                    frame as isize * HOP as isize + offset as isize - (FFT_SIZE / 2) as isize;
                let reflected = if position < 0 {
                    (-position) as usize
                } else if position >= SAMPLES as isize {
                    2 * SAMPLES - position as usize - 2
                } else {
                    position as usize
                };
                Complex {
                    re: padded[reflected] * hann[offset],
                    im: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let spectrum = fft(&window);
        let power = spectrum[..=FFT_SIZE / 2]
            .iter()
            .map(|value| value.re * value.re + value.im * value.im)
            .collect::<Vec<_>>();
        for (mel, filter) in filters.iter().enumerate() {
            let value = filter
                .iter()
                .zip(&power)
                .map(|(weight, power)| weight * power)
                .sum::<f32>();
            output[mel * FRAMES + frame] = value.max(1e-10).log10();
        }
    }
    let floor = output.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for value in &mut output {
        *value = (value.max(floor) + 4.0) / 4.0;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(data: &[i16], channels: u16) -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend((36 + data.len() as u32 * 2).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16_u32.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(channels.to_le_bytes());
        bytes.extend(16000_u32.to_le_bytes());
        bytes.extend((32000_u32 * u32::from(channels)).to_le_bytes());
        bytes.extend((2 * channels).to_le_bytes());
        bytes.extend(16_u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend((data.len() as u32 * 2).to_le_bytes());
        for sample in data {
            bytes.extend(sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn stereo_pcm_is_mixed_and_bad_headers_are_rejected() {
        let audio = decode_wav(&wav(&[32767, -32767, 16384, 16384], 2)).unwrap();
        assert_eq!(audio, [0.0, 0.5]);
        let mut truncated = wav(&[1, 2], 1);
        truncated.pop();
        assert!(decode_wav(&truncated).is_err());
        let mut rate = wav(&[1, 2], 1);
        rate[24..28].copy_from_slice(&44100_u32.to_le_bytes());
        assert!(decode_wav(&rate).is_err());
    }

    fn formatted_wav(format: &[u8], data: &[u8]) -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend((20 + format.len() as u32 + data.len() as u32).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend((format.len() as u32).to_le_bytes());
        bytes.extend(format);
        bytes.extend(b"data");
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn extensible_wav_accepts_pcm_and_float_guids_but_rejects_unknown_formats() {
        for (encoding, bits, data) in [
            (1u16, 16u16, 16384i16.to_le_bytes().to_vec()),
            (3u16, 32u16, 0.5f32.to_le_bytes().to_vec()),
        ] {
            let mut format = Vec::new();
            format.extend(0xfffeu16.to_le_bytes());
            format.extend(1u16.to_le_bytes());
            format.extend(16000u32.to_le_bytes());
            format.extend((16000u32 * u32::from(bits / 8)).to_le_bytes());
            format.extend((bits / 8).to_le_bytes());
            format.extend(bits.to_le_bytes());
            format.extend(22u16.to_le_bytes());
            format.extend(bits.to_le_bytes());
            format.extend(4u32.to_le_bytes()); // mono speaker mask
            format.extend(u32::from(encoding).to_le_bytes());
            format.extend([0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
            assert_eq!(decode_wav(&formatted_wav(&format, &data)).unwrap(), [0.5]);
            format[39] ^= 1;
            assert!(decode_wav(&formatted_wav(&format, &data)).is_err());
        }
    }

    #[test]
    fn floating_wav_clips_finite_samples_before_mixing_and_rejects_nonfinite_values() {
        let mut format = Vec::new();
        format.extend(3u16.to_le_bytes());
        format.extend(2u16.to_le_bytes());
        format.extend(16000u32.to_le_bytes());
        format.extend(128000u32.to_le_bytes());
        format.extend(8u16.to_le_bytes());
        format.extend(32u16.to_le_bytes());
        let samples = [f32::MAX, f32::MAX, -2.0, 0.5];
        let data = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            decode_wav(&formatted_wav(&format, &data)).unwrap(),
            [1.0, -0.25]
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let data = [value, 0.0]
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>();
            assert!(decode_wav(&formatted_wav(&format, &data)).is_err());
        }
    }

    #[test]
    fn mixed_radix_fft_matches_direct_dft() {
        let input = (0..20)
            .map(|i| Complex {
                re: (i as f32).sin(),
                im: 0.0,
            })
            .collect::<Vec<_>>();
        let actual = fft(&input);
        for (k, value) in actual.iter().enumerate() {
            let reference = input
                .iter()
                .enumerate()
                .fold(Complex::default(), |mut sum, (n, x)| {
                    let angle = -std::f32::consts::TAU * (k * n) as f32 / input.len() as f32;
                    sum.re += x.re * angle.cos();
                    sum.im += x.re * angle.sin();
                    sum
                });
            assert!((value.re - reference.re).abs() < 1e-4);
            assert!((value.im - reference.im).abs() < 1e-4);
        }
    }

    #[test]
    fn slaney_filter_bank_is_nonnegative_and_covers_speech() {
        let filters = mel_filters();
        assert_eq!(filters.len(), 80);
        assert!(
            filters.iter().all(|row| row.len() == 201
                && row.iter().all(|value| value.is_finite() && *value >= 0.0))
        );
        assert!(
            filters
                .iter()
                .all(|row| row.iter().any(|value| *value > 0.0))
        );
    }

    #[test]
    fn feature_extraction_rejects_oversize_and_observes_cancellation() {
        assert!(log_mel(&vec![0.0; SAMPLES + 1], &AtomicBool::new(false)).is_err());
        assert!(log_mel(&[0.0], &AtomicBool::new(true)).is_err());
    }
}
