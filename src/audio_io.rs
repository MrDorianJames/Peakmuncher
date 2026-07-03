//! Audio file loading and saving for WAV, FLAC, and MP3.
//!
//! All audio is held internally as interleaved f32 in [-1, 1] with a known
//! sample rate and channel count.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Arc<Vec<f32>>, // shared, cheap to clone
    pub sample_rate: u32,
    pub channels: u16,
    /// Bit depth of the source file. Used to preserve the original depth
    /// on save (e.g. load 24-bit WAV → save 24-bit WAV by default).
    /// Defaults to 16 for synthetic / decoded buffers (MP3, etc.).
    pub source_bit_depth: u16,
}

impl AudioBuffer {
    pub fn frames(&self) -> usize {
        let ch = self.channels.max(1) as usize;
        self.samples.len() / ch
    }

    pub fn duration_secs(&self) -> f32 {
        let sr = self.sample_rate.max(1);
        self.frames() as f32 / sr as f32
    }

    /// Mono mix-down for waveform display.
    pub fn to_mono(&self) -> Vec<f32> {
        let ch = self.channels.max(1) as usize;
        if ch == 1 {
            return (*self.samples).clone();
        }
        let frames = self.frames();
        let mut out = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut acc = 0.0;
            for c in 0..ch {
                acc += self.samples[f * ch + c];
            }
            out.push(acc / ch as f32);
        }
        out
    }
}

pub fn load(path: &Path) -> Result<AudioBuffer> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => load_wav(path),
        "flac" => load_flac(path),
        "mp3" => load_mp3(path),
        "aiff" | "aif" | "aifc" => load_aiff(path),
        other => Err(anyhow!("Unsupported input format: .{other}")),
    }
}

pub fn save(path: &Path, buf: &AudioBuffer) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => save_wav(path, buf),
        "flac" => save_flac(path, buf),
        "mp3" => save_mp3(path, buf),
        "aiff" | "aif" => save_aiff(path, buf),
        other => Err(anyhow!("Unsupported output format: .{other}")),
    }
}

// ---- WAV ----------------------------------------------------------------

fn load_wav(path: &Path) -> Result<AudioBuffer> {
    let mut reader = hound::WavReader::open(path).context("opening WAV")?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(AudioBuffer {
        samples: Arc::new(samples),
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        source_bit_depth: spec.bits_per_sample,
    })
}

fn save_wav(path: &Path, buf: &AudioBuffer) -> Result<()> {
    // Match the source bit depth where possible. Common DAW outputs are
    // 16 or 24-bit integer; 32-bit float is preserved when the source
    // already was float-format, or as a fallback for unusual depths.
    let (bits, sample_format) = match buf.source_bit_depth {
        16 => (16, hound::SampleFormat::Int),
        24 => (24, hound::SampleFormat::Int),
        32 => (32, hound::SampleFormat::Float),
        _ => (24, hound::SampleFormat::Int), // safe default for 8/20/anything else
    };
    let spec = hound::WavSpec {
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        bits_per_sample: bits,
        sample_format,
    };
    let mut w = hound::WavWriter::create(path, spec).context("creating WAV")?;
    match sample_format {
        hound::SampleFormat::Float => {
            for &s in buf.samples.iter() {
                w.write_sample(s)?;
            }
        }
        hound::SampleFormat::Int => {
            // Scale clamped float to integer of `bits` width. Use i32 as the
            // common write type — hound packs to the requested bit depth.
            let max = (1i64 << (bits - 1)) as f32;
            let upper = (max - 1.0) as i32;
            let lower = -(max as i32);
            for &s in buf.samples.iter() {
                let v = (s.clamp(-1.0, 1.0) * max) as i32;
                w.write_sample(v.clamp(lower, upper))?;
            }
        }
    }
    w.finalize()?;
    Ok(())
}

// ---- FLAC ---------------------------------------------------------------

fn load_flac(path: &Path) -> Result<AudioBuffer> {
    let mut reader = claxon::FlacReader::open(path).context("opening FLAC")?;
    let info = reader.streaminfo();
    let bits = info.bits_per_sample as u32;
    let max = (1i64 << (bits - 1)) as f32;
    let samples: Vec<f32> = reader
        .samples()
        .map(|s| s.map(|v| v as f32 / max))
        .collect::<Result<_, _>>()?;
    Ok(AudioBuffer {
        samples: Arc::new(samples),
        sample_rate: info.sample_rate,
        channels: info.channels as u16,
        source_bit_depth: info.bits_per_sample as u16,
    })
}

fn save_flac(path: &Path, buf: &AudioBuffer) -> Result<()> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    // Match source bit depth where possible. FLAC supports 8/16/24-bit
    // integer (and 20-bit, but rarely seen). 32 falls back to 24 since
    // FLAC's max is 24-bit per the streamable subset.
    let bits_per_sample = match buf.source_bit_depth {
        16 => 16usize,
        24 | 32 => 24,
        _ => 24,
    };
    let scale = ((1i32 << (bits_per_sample - 1)) - 1) as f32;
    let pcm: Vec<i32> = buf
        .samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * scale) as i32)
        .collect();

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_c, e)| anyhow!("flacenc bad config: {e:?}"))?;

    let source = MemSource::from_samples(
        &pcm,
        buf.channels as usize,
        bits_per_sample,
        buf.sample_rate as usize,
    );

    let stream = flacenc::encode_with_fixed_block_size(
        &config,
        source,
        config.block_size,
    )
    .map_err(|e| anyhow!("flacenc encode failed: {e:?}"))?;

    // Serialize the encoded stream to a byte buffer, then write to disk.
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| anyhow!("flacenc serialize failed: {e:?}"))?;
    std::fs::write(path, sink.as_slice()).context("writing FLAC file")?;
    Ok(())
}

// ---- MP3 ----------------------------------------------------------------

fn load_mp3(path: &Path) -> Result<AudioBuffer> {
    let file = std::fs::File::open(path).context("opening MP3")?;
    let mut decoder = minimp3::Decoder::new(file);
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                let fr_sr = frame.sample_rate as u32;
                let fr_ch = frame.channels as u16;
                if sample_rate == 0 {
                    sample_rate = fr_sr;
                    channels = fr_ch;
                } else if fr_sr != sample_rate || fr_ch != channels {
                    // Mid-stream format change — skip this frame to keep
                    // the buffer interleaved consistently.
                    continue;
                }
                samples.extend(frame.data.into_iter().map(|s| s as f32 / i16::MAX as f32));
            }
            Err(minimp3::Error::Eof) => break,
            Err(minimp3::Error::SkippedData) => continue,
            Err(minimp3::Error::InsufficientData) => continue,
            Err(e) => return Err(anyhow!("MP3 decode error: {e:?}")),
        }
    }
    if sample_rate == 0 || channels == 0 || samples.is_empty() {
        return Err(anyhow!("MP3 contains no decodable audio frames"));
    }
    Ok(AudioBuffer {
        samples: Arc::new(samples),
        sample_rate,
        channels,
        source_bit_depth: 16, // MP3 is lossy; depth not meaningful
    })
}

fn save_mp3(path: &Path, buf: &AudioBuffer) -> Result<()> {
    use mp3lame_encoder::{Builder, FlushNoGap, InterleavedPcm, MonoPcm};
    let mut builder = Builder::new().ok_or_else(|| anyhow!("LAME builder failed"))?;
    builder
        .set_num_channels(buf.channels as u8)
        .map_err(|e| anyhow!("{e:?}"))?;
    builder
        .set_sample_rate(buf.sample_rate)
        .map_err(|e| anyhow!("{e:?}"))?;
    builder
        .set_brate(mp3lame_encoder::Bitrate::Kbps192)
        .map_err(|e| anyhow!("{e:?}"))?;
    builder
        .set_quality(mp3lame_encoder::Quality::Good)
        .map_err(|e| anyhow!("{e:?}"))?;
    let mut enc = builder.build().map_err(|e| anyhow!("{e:?}"))?;

    // f32 [-1,1] -> i16
    let pcm: Vec<i16> = buf
        .samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    // The encoder writes into uninitialized memory, so we keep the buffer as
    // Vec<MaybeUninit<u8>> until we know how many bytes were actually written,
    // then convert to a Vec<u8> via the reported length.
    use std::mem::MaybeUninit;
    let cap = mp3lame_encoder::max_required_buffer_size(pcm.len());
    let mut out: Vec<MaybeUninit<u8>> = Vec::with_capacity(cap);
    // SAFETY: we only ever read [..written], and written is what the encoder reports.
    unsafe { out.set_len(cap); }

    let n = if buf.channels == 1 {
        enc.encode(MonoPcm(&pcm), &mut out[..])
            .map_err(|e| anyhow!("{e:?}"))?
    } else {
        enc.encode(InterleavedPcm(&pcm), &mut out[..])
            .map_err(|e| anyhow!("{e:?}"))?
    };
    let mut written = n;
    let tail = enc
        .flush::<FlushNoGap>(&mut out[written..])
        .map_err(|e| anyhow!("{e:?}"))?;
    written += tail;

    // Convert the initialized prefix to a regular Vec<u8>.
    let bytes: Vec<u8> = out[..written]
        .iter()
        .map(|m| unsafe { m.assume_init() })
        .collect();
    std::fs::write(path, &bytes).context("writing MP3 file")?;
    Ok(())
}

// ---- AIFF / AIFF-C ------------------------------------------------------
//
// AIFF is an IFF chunked container with big-endian byte order. The format
// has two flavors:
//   - Classic AIFF (FORM type "AIFF"): always uncompressed big-endian PCM.
//   - AIFF-C        (FORM type "AIFC"): introduces a compression code in
//     the COMM chunk. The vast majority of AIFC files in the wild are
//     uncompressed too — they just use compression codes "NONE" (BE PCM,
//     identical to classic AIFF), "sowt" (LE PCM, written by some Apple
//     tools), "fl32" (32-bit BE float) or "fl64" (64-bit BE float).
//     Lossy variants (ima4, ulaw, MAC3, etc.) are rejected with a clear
//     error rather than silently producing garbage.
//
// Spec references:
//   - AIFF 1.3:  https://muratnkonar.com/aiff/index.html
//   - AIFF-C:    Apple's IM:Sound Manager (1991)
//
// Implementation is intentionally focused: chunk walker + COMM parser + a
// per-format SSND decoder. ~150 lines beats a 50KB dependency.

use std::io::{Read, Write};

fn load_aiff(path: &Path) -> Result<AudioBuffer> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .context("opening AIFF")?
        .read_to_end(&mut bytes)
        .context("reading AIFF")?;
    if bytes.len() < 12 || &bytes[..4] != b"FORM" {
        return Err(anyhow!("Not a valid AIFF file (missing FORM header)"));
    }
    let form_type = &bytes[8..12];
    let is_aifc = match form_type {
        b"AIFF" => false,
        b"AIFC" => true,
        other => {
            return Err(anyhow!(
                "Unsupported FORM type: {}",
                std::str::from_utf8(other).unwrap_or("?")
            ));
        }
    };

    // Walk top-level chunks looking for COMM and SSND.
    let mut pos = 12usize;
    let mut comm: Option<&[u8]> = None;
    let mut ssnd: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = be_u32(&bytes[pos + 4..pos + 8]) as usize;
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(size)
            .ok_or_else(|| anyhow!("AIFF: chunk size overflow"))?;
        if data_end > bytes.len() {
            return Err(anyhow!("AIFF: chunk extends past end of file"));
        }
        match id {
            b"COMM" => comm = Some(&bytes[data_start..data_end]),
            b"SSND" => ssnd = Some(&bytes[data_start..data_end]),
            _ => {}
        }
        // Chunks pad to even byte alignment.
        pos = data_end + (size & 1);
    }
    let comm = comm.ok_or_else(|| anyhow!("AIFF: missing COMM chunk"))?;
    let ssnd = ssnd.ok_or_else(|| anyhow!("AIFF: missing SSND chunk"))?;

    if comm.len() < 18 {
        return Err(anyhow!("AIFF: COMM chunk too small"));
    }
    let channels = be_u16(&comm[0..2]) as u16;
    let _num_frames = be_u32(&comm[2..6]);
    let bits_per_sample = be_u16(&comm[6..8]) as u16;
    let sample_rate = parse_extended_f80(&comm[8..18]) as u32;

    // AIFC adds a 4-byte compression type and a Pascal string.
    let compression: [u8; 4] = if is_aifc {
        if comm.len() < 22 {
            return Err(anyhow!("AIFC: COMM chunk truncated"));
        }
        let mut c = [0u8; 4];
        c.copy_from_slice(&comm[18..22]);
        c
    } else {
        *b"NONE"
    };

    // SSND chunk: 4-byte offset, 4-byte block size, then the audio bytes.
    if ssnd.len() < 8 {
        return Err(anyhow!("AIFF: SSND chunk too small"));
    }
    let offset = be_u32(&ssnd[0..4]) as usize;
    let audio_bytes = ssnd
        .get(8 + offset..)
        .ok_or_else(|| anyhow!("AIFF: SSND offset past end"))?;

    let samples = decode_aiff_samples(audio_bytes, &compression, bits_per_sample)?;

    Ok(AudioBuffer {
        samples: Arc::new(samples),
        sample_rate,
        channels,
        source_bit_depth: bits_per_sample,
    })
}

fn decode_aiff_samples(bytes: &[u8], compression: &[u8; 4], bits: u16) -> Result<Vec<f32>> {
    match compression {
        // Big-endian integer PCM (classic AIFF and AIFC "NONE").
        b"NONE" | b"twos" | b"in16" | b"in24" | b"in32" => {
            decode_pcm_int(bytes, bits, /*little_endian=*/ false)
        }
        // Apple's little-endian integer PCM.
        b"sowt" | b"SOWT" => decode_pcm_int(bytes, bits, /*little_endian=*/ true),
        // 32-bit big-endian float.
        b"fl32" | b"FL32" => {
            if bits != 32 {
                return Err(anyhow!(
                    "AIFC fl32 with {} bits per sample (expected 32)",
                    bits
                ));
            }
            let n = bytes.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = f32::from_be_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                out.push(v);
            }
            Ok(out)
        }
        // 64-bit big-endian float.
        b"fl64" | b"FL64" => {
            if bits != 64 {
                return Err(anyhow!(
                    "AIFC fl64 with {} bits per sample (expected 64)",
                    bits
                ));
            }
            let n = bytes.len() / 8;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                out.push(f64::from_be_bytes(buf) as f32);
            }
            Ok(out)
        }
        other => Err(anyhow!(
            "Unsupported AIFC compression: {} (PeakMuncher supports NONE/sowt/fl32/fl64; \
             lossy variants like ima4/ulaw/MAC3 aren't supported — convert to WAV first)",
            std::str::from_utf8(other).unwrap_or("?")
        )),
    }
}

fn decode_pcm_int(bytes: &[u8], bits: u16, little_endian: bool) -> Result<Vec<f32>> {
    let bytes_per_sample = ((bits + 7) / 8) as usize;
    if bytes_per_sample == 0 || bytes_per_sample > 4 {
        return Err(anyhow!("AIFF: unsupported {} bits per sample", bits));
    }
    let max = (1i64 << (bits - 1)) as f32;
    let n = bytes.len() / bytes_per_sample;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = &bytes[i * bytes_per_sample..(i + 1) * bytes_per_sample];
        // Sign-extend variable-width integer to i32, in the requested
        // endianness.
        let mut v: i32 = 0;
        if little_endian {
            for (j, &b) in chunk.iter().enumerate() {
                v |= (b as i32) << (8 * j);
            }
        } else {
            for &b in chunk.iter() {
                v = (v << 8) | (b as i32);
            }
        }
        // Sign-extend from `bits` to 32.
        let shift = 32 - bits as i32;
        v = (v << shift) >> shift;
        out.push(v as f32 / max);
    }
    Ok(out)
}

/// Parse IEEE 754 80-bit extended precision (used by AIFF for sample rate).
/// We only need the integer part of common rates (44100, 48000, 96000, …)
/// so a full IEEE-extended decoder isn't required, but doing it properly is
/// easy enough.
fn parse_extended_f80(b: &[u8]) -> f64 {
    let sign = (b[0] >> 7) as u64;
    let exponent = (((b[0] as u64) & 0x7f) << 8) | (b[1] as u64);
    let mut mantissa: u64 = 0;
    for i in 0..8 {
        mantissa = (mantissa << 8) | (b[2 + i] as u64);
    }
    if exponent == 0 && mantissa == 0 {
        return 0.0;
    }
    if exponent == 0x7fff {
        return f64::INFINITY;
    }
    let exp = exponent as i32 - 16383 - 63;
    let value = mantissa as f64 * 2f64.powi(exp);
    if sign == 1 {
        -value
    } else {
        value
    }
}

fn be_u16(b: &[u8]) -> u16 {
    ((b[0] as u16) << 8) | (b[1] as u16)
}
fn be_u32(b: &[u8]) -> u32 {
    ((b[0] as u32) << 24) | ((b[1] as u32) << 16) | ((b[2] as u32) << 8) | (b[3] as u32)
}

/// Encode an f64 sample rate as an 80-bit IEEE 754 extended-precision
/// big-endian, the format AIFF's COMM chunk requires.
fn write_extended_f80(rate: f64) -> [u8; 10] {
    let mut out = [0u8; 10];
    if rate == 0.0 {
        return out;
    }
    let sign = if rate < 0.0 { 1u8 } else { 0u8 };
    let v = rate.abs();
    let exp_unbiased = v.log2().floor() as i32;
    let exponent = (exp_unbiased + 16383) as u16;
    let mantissa = (v / 2f64.powi(exp_unbiased) * 2f64.powi(63)) as u64;
    out[0] = (sign << 7) | ((exponent >> 8) as u8 & 0x7f);
    out[1] = exponent as u8;
    out[2..10].copy_from_slice(&mantissa.to_be_bytes());
    out
}

fn save_aiff(path: &Path, buf: &AudioBuffer) -> Result<()> {
    // Match source bit depth where possible. AIFF supports 8/16/24/32-bit
    // big-endian integer in classic flavor. Most DAWs accept 16 and 24
    // without issue; 32-bit float requires AIFC and is omitted here.
    let bits: u16 = match buf.source_bit_depth {
        16 => 16,
        24 | 32 => 24, // 32-bit float source folds into 24-bit int on save
        _ => 24,
    };
    let bytes_per_sample: usize = ((bits + 7) / 8) as usize;
    let channels = buf.channels.max(1);
    let frames = buf.frames();
    let audio_bytes_len = frames * channels as usize * bytes_per_sample;

    // Encode as big-endian integer of `bits` width.
    let mut audio = Vec::with_capacity(audio_bytes_len);
    let max = (1i64 << (bits - 1)) as f32;
    let upper = (max - 1.0) as i32;
    let lower = -(max as i32);
    for &s in buf.samples.iter() {
        let v = ((s.clamp(-1.0, 1.0) * max) as i32).clamp(lower, upper);
        // Write the lowest `bytes_per_sample` bytes of `v` in big-endian
        // order. e.g. 24-bit takes the top 3 bytes of the i32 BE form.
        let be = v.to_be_bytes();
        audio.extend_from_slice(&be[4 - bytes_per_sample..]);
    }

    // COMM chunk: channels(2), numSampleFrames(4), sampleSize(2), sampleRate(10) = 18 bytes
    let mut comm = Vec::with_capacity(18);
    comm.extend_from_slice(&(channels as u16).to_be_bytes());
    comm.extend_from_slice(&(frames as u32).to_be_bytes());
    comm.extend_from_slice(&bits.to_be_bytes());
    comm.extend_from_slice(&write_extended_f80(buf.sample_rate as f64));

    // SSND chunk: offset(4) + blockSize(4) + audio
    let mut ssnd = Vec::with_capacity(8 + audio.len());
    ssnd.extend_from_slice(&0u32.to_be_bytes()); // offset
    ssnd.extend_from_slice(&0u32.to_be_bytes()); // blockSize
    ssnd.extend_from_slice(&audio);

    // FORM chunk wraps everything. Total size = 4 (form_type) + each
    // chunk's (8-byte header + body, padded to even).
    let comm_size = comm.len() as u32;
    let ssnd_size = ssnd.len() as u32;
    let total = 4 + 8 + comm.len() + (comm.len() & 1) + 8 + ssnd.len() + (ssnd.len() & 1);

    let mut out: Vec<u8> = Vec::with_capacity(8 + total);
    out.extend_from_slice(b"FORM");
    out.extend_from_slice(&(total as u32).to_be_bytes());
    out.extend_from_slice(b"AIFF");
    out.extend_from_slice(b"COMM");
    out.extend_from_slice(&comm_size.to_be_bytes());
    out.extend_from_slice(&comm);
    if comm.len() & 1 == 1 {
        out.push(0);
    }
    out.extend_from_slice(b"SSND");
    out.extend_from_slice(&ssnd_size.to_be_bytes());
    out.extend_from_slice(&ssnd);
    if ssnd.len() & 1 == 1 {
        out.push(0);
    }

    let mut f = std::fs::File::create(path).context("creating AIFF")?;
    f.write_all(&out).context("writing AIFF")?;
    Ok(())
}
