//! EVK4 Playback Server
//!
//! A unified MJPEG server that accepts file playback requests via HTTP API
//! and renders them into a live MJPEG stream. Supports:
//!
//!   - `.raw`  — raw EVT3 byte stream decoded via neuromorphic_drivers adapter
//!   - `.mp4` / `.h264` — video decoded via ffmpeg piped as raw grayscale frames
//!
//! The MJPEG output stream is always running. The input source switches
//! atomically when a new /play request arrives.
//!
//! HTTP API
//! --------
//!   POST /play   { "file": "/path/to/recording.raw", "speed": 1.0 }
//!   POST /stop
//!   GET  /status → { "playing": bool, "file": str|null, "type": str|null }
//!   GET  /        → MJPEG stream (multipart/x-mixed-replace)
//!
//! Usage:
//!   cargo run --release playback_server -- --width 1280 --height 720

use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use image::codecs::jpeg::JpegEncoder;
use image::{GrayImage};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug, Clone)]
#[command(name = "playback-server", about = "Unified EVK4 playback MJPEG server")]
struct Args {
    /// Sensor / frame width in pixels
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Sensor / frame height in pixels
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// MJPEG output frame rate
    #[arg(long, default_value_t = 60)]
    fps: u64,

    /// JPEG quality (1–100)
    #[arg(long, default_value_t = 50)]
    quality: u8,

    /// HTTP bind address for both MJPEG stream and API
    #[arg(long, default_value = "0.0.0.0:8084")]
    bind: String,

    /// Recording_Dir
    #[arg(long, default_value = "/home/eventide/recordings")]
    recordings: String,
}

// ── Shared state ──────────────────────────────────────────────────────────────

/// The pixel buffer and playback status shared between the decoder thread,
/// encoder thread, and HTTP API handler.
struct State {
    /// Grayscale pixel buffer — width * height bytes, mid-grey (127) at rest.
    pixels: Vec<u8>,
    /// Most recently JPEG-encoded frame ready for HTTP clients.
    latest_jpeg: Option<Vec<u8>>,
    /// Currently active playback request, if any.
    active: Option<PlayRequest>,
    /// Signal to the decoder thread to stop current playback.
    stop_signal: bool,
}

#[derive(Clone, Debug)]
struct PlayRequest {
    file: PathBuf,
    speed: f64,
}

// ── MJPEG helpers ─────────────────────────────────────────────────────────────

const BOUNDARY: &str = "evk4playback";

fn encode_jpeg(pixels: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let img = GrayImage::from_raw(width, height, pixels.to_vec())
        .expect("pixel buffer size mismatch");
    let mut buf = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut buf, quality)
        .encode_image(&img)
        .expect("JPEG encode failed");
    buf.into_inner()
}

fn mjpeg_part(jpeg: &[u8]) -> Vec<u8> {
    let mut part = format!(
        "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
        jpeg.len()
    )
    .into_bytes();
    part.extend_from_slice(jpeg);
    part.extend_from_slice(b"\r\n");
    part
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

/// Parse the first line method and path from a raw HTTP request buffer.
fn parse_request(buf: &[u8]) -> (String, String, String) {
    let s = std::str::from_utf8(buf).unwrap_or("");
    let first = s.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path   = parts.next().unwrap_or("/").to_string();

    // Extract body after blank line separator
    let body = if let Some(idx) = s.find("\r\n\r\n") {
        s[idx + 4..].trim_end_matches('\0').to_string()
    } else if let Some(idx) = s.find("\n\n") {
        s[idx + 2..].trim_end_matches('\0').to_string()
    } else {
        String::new()
    };

    (method, path, body)
}

fn respond_json(stream: &mut TcpStream, status: &str, body: &str) {
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    );
}

// ── EVT3 decoder ──────────────────────────────────────────────────────────────

/// Decode a `.raw` EVT3 file into the shared pixel buffer, pacing at `speed`.
fn run_evt3_decoder(
    request: PlayRequest,
    width: u32,
    height: u32,
    state: Arc<Mutex<State>>,
) {
    use std::time::{Duration, Instant};

    let mut adapter =
        neuromorphic_drivers::adapters::evt3::Adapter::from_dimensions(width.try_into().unwrap(), height.try_into().unwrap());

    let mut file = match std::fs::File::open(&request.file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[evt3] Failed to open {:?}: {e}", request.file);
            finish_playback(&state);
            return;
        }
    };

    let mut chunk = vec![0u8; 131072];
    let mut pacer_anchor: Option<(u64, Instant)> = None;

    loop {
        // Check stop signal
        if state.lock().unwrap().stop_signal {
            break;
        }

        let n = match file.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => { eprintln!("[evt3] Read error: {e}"); break; }
        };

        let mut dvs_last_t: Option<u64> = None;
        let mut trigger_last_t: Option<u64> = None;

        {
            let mut st = state.lock().unwrap();
            if st.stop_signal { break; }

            adapter.convert(
                &chunk[..n],
                |dvs_event| {
                    let x = dvs_event.x as u32;
                    let y = dvs_event.y as u32;
                    if x < width && y < height {
                        st.pixels[y as usize * width as usize + x as usize] =
                            if dvs_event.polarity as u8 != 0 { 255 } else { 0 };
                    }
                    dvs_last_t = Some(dvs_event.t);
                },
                |trigger_event| {
                    trigger_last_t = Some(trigger_event.t);
                },
            );
        }

        // Real-time pacing against sensor timestamps
        let chunk_last_t = std::cmp::max(dvs_last_t, trigger_last_t);
        if let Some(last_t) = chunk_last_t {
            let (origin_t, origin_wall) = *pacer_anchor.get_or_insert_with(|| {
                eprintln!("[evt3] Pacer anchored at t={last_t} µs");
                (last_t, Instant::now())
            });

            if last_t > origin_t {
                let sensor_delta = last_t - origin_t;
                let wall_target_us = (sensor_delta as f64 / request.speed) as u64;
                let wall_target = origin_wall + Duration::from_micros(wall_target_us);
                let now = Instant::now();
                if wall_target > now {
                    thread::sleep(wall_target - now);
                }
            }
        }
    }

    finish_playback(&state);
    eprintln!("[evt3] Playback complete.");
}

// ── Video decoder (ffmpeg) ────────────────────────────────────────────────────

/// Decode an mp4/h264 file by piping raw grayscale frames from ffmpeg.
fn run_video_decoder(
    request: PlayRequest,
    width: u32,
    height: u32,
    state: Arc<Mutex<State>>,
) {
    // Build ffmpeg command — output raw grayscale frames at native rate,
    // scaled to sensor dimensions if necessary.
    // `atempo` / `setpts` handle speed adjustment.
    let setpts = format!("setpts={:.6}/TB", 1.0 / request.speed);
    let vf = format!(
        "scale={}:{},format=gray,{}",
        width, height, setpts
    );

    let mut child: Child = match Command::new("ffmpeg")
        .args([
            "-re",                          // read at native frame rate
            "-i", request.file.to_str().unwrap_or(""),
            "-vf", &vf,
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "pipe:1",                       // output to stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[video] Failed to spawn ffmpeg: {e}");
            finish_playback(&state);
            return;
        }
    };

    let frame_size = (width * height) as usize;
    let mut frame_buf = vec![0u8; frame_size];
    let mut stdout = child.stdout.take().expect("ffmpeg stdout");

    loop {
        if state.lock().unwrap().stop_signal {
            break;
        }

        // Read exactly one raw grayscale frame from ffmpeg
        match read_exact_from(&mut stdout, &mut frame_buf) {
            Ok(false) => break, // EOF
            Err(e) => { eprintln!("[video] ffmpeg read error: {e}"); break; }
            Ok(true) => {}
        }

        // Write the frame into the shared pixel buffer
        {
            let mut st = state.lock().unwrap();
            if st.stop_signal { break; }
            st.pixels.copy_from_slice(&frame_buf);
        }
    }

    let _ = child.kill();
    finish_playback(&state);
    eprintln!("[video] Playback complete.");
}

/// Read exactly buf.len() bytes, returning Ok(false) on clean EOF.
fn read_exact_from<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut total = 0;
    while total < buf.len() {
        match reader.read(&mut buf[total..]) {
            Ok(0) => return Ok(false),
            Ok(n) => total += n,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn finish_playback(state: &Arc<Mutex<State>>) {
    let mut st = state.lock().unwrap();
    st.active = None;
    st.stop_signal = false;
    st.pixels.fill(127); // reset to mid-grey
}

// ── Decoder dispatch ──────────────────────────────────────────────────────────

fn spawn_decoder(request: PlayRequest, width: u32, height: u32, state: Arc<Mutex<State>>) {
    let ext = request
        .file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "raw" => {
            thread::spawn(move || run_evt3_decoder(request, width, height, state));
        }
        "mp4" | "h264" | "mkv" | "avi" => {
            thread::spawn(move || run_video_decoder(request, width, height, state));
        }
        other => {
            eprintln!("[playback] Unsupported file type: .{other}");
            finish_playback(&state);
        }
    }
}

// ── HTTP client handler ───────────────────────────────────────────────────────

fn handle_client(
    mut stream: TcpStream,
    state: Arc<Mutex<State>>,
    args: Arc<Args>,
) {
    let mut req_buf = [0u8; 4096];
    let n = match stream.read(&mut req_buf) {
        Ok(n) => n,
        Err(_) => return,
    };

    let (method, path, body) = parse_request(&req_buf[..n]);

    match (method.as_str(), path.as_str()) {

        // ── MJPEG stream ──────────────────────────────────────────────────────
        ("GET", "/") | ("GET", "/stream") => {
            let headers = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: multipart/x-mixed-replace; boundary={BOUNDARY}\r\n\
                 Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
            );
            if stream.write_all(headers.as_bytes()).is_err() { return; }

            let frame_interval = Duration::from_nanos(1_000_000_000 / args.fps);
            let mut last_sent = Instant::now();

            loop {
                let elapsed = last_sent.elapsed();
                if elapsed < frame_interval {
                    thread::sleep(frame_interval - elapsed);
                }
                let frame = state.lock().unwrap().latest_jpeg.clone();
                if let Some(jpeg) = frame {
                    if stream.write_all(&mjpeg_part(&jpeg)).is_err() { break; }
                    last_sent = Instant::now();
                }
            }
        }

        // ── POST /play ────────────────────────────────────────────────────────
        ("POST", "/play") => {
            // Parse JSON body: { "file": "...", "speed": 1.0 }
            let cam_val = json_str_field(&body, "cam");
            let file_val = json_str_field(&body, "filename");
            let speed_val = json_f64_field(&body, "speed").unwrap_or(1.0);

            let file = match file_val {
                Some(f) if !f.is_empty() => PathBuf::from(f),
                _ => {
                    respond_json(&mut stream, "400 Bad Request",
                        r#"{"error":"missing field: file"}"#);
                    return;
                }
            };

            let filepath = match cam_val {
                Some("picam") => PathBuf::from(format!("{}/picam/{}", args.recordings, file.display())),
                Some("ircam") => PathBuf::from(format!("{}/ircam/{}", args.recordings, file.display())),
                Some("evk") => PathBuf::from(format!("{}/evk/{}", args.recordings, file.display())),
                _ => {
                    respond_json(&mut stream, "404 Not Found",
                    r#"{"error":"Invalid Camera Type or no camera supplied"}"#);
                    return
                    }
            };

            //println!("{}", filepath.display());
            
            if !filepath.exists() {
                respond_json(&mut stream, "404 Not Found",
                    r#"{"error":"file not found"}"#);
                return;
            }


            let request = PlayRequest { file: filepath.clone(), speed: speed_val };

            {
                let mut st = state.lock().unwrap();
                // Signal any running decoder to stop
                st.stop_signal = true;
                st.active = Some(request.clone());
            }

            // Brief pause to let any running decoder observe the stop signal
            // before we clear it and spawn the new decoder.
            thread::sleep(Duration::from_millis(50));
            state.lock().unwrap().stop_signal = false;

            spawn_decoder(request, args.width, args.height, Arc::clone(&state));

            let resp = format!(
                r#"{{"ok":true,"file":"{}","speed":{}}}"#,
                file.display(), speed_val
            );
            respond_json(&mut stream, "200 OK", &resp);
        }

        // ── POST /stop ────────────────────────────────────────────────────────
        ("POST", "/stop") => {
            state.lock().unwrap().stop_signal = true;
            respond_json(&mut stream, "200 OK", r#"{"ok":true}"#);
        }

        // ── GET /status ───────────────────────────────────────────────────────
        ("GET", "/status") => {
            let st = state.lock().unwrap();
            let body = match &st.active {
                Some(req) => format!(
                    r#"{{"playing":true,"file":"{}","type":"{}"}}"#,
                    req.file.display(),
                    req.file.extension().and_then(|e| e.to_str()).unwrap_or("unknown")
                ),
                None => r#"{"playing":false,"file":null,"type":null}"#.to_string(),
            };
            drop(st);
            respond_json(&mut stream, "200 OK", &body);
        }

        // ── CORS preflight ────────────────────────────────────────────────────
        ("OPTIONS", _) => {
            let _ = stream.write_all(
                b"HTTP/1.1 204 No Content\r\n\
                  Access-Control-Allow-Origin: *\r\n\
                  Access-Control-Allow-Methods: GET, POST\r\n\
                  Access-Control-Allow-Headers: Content-Type\r\n\r\n",
            );
        }

        _ => {
            respond_json(&mut stream, "404 Not Found", r#"{"error":"not found"}"#);
        }
    }
}

// ── Minimal JSON field extractors ─────────────────────────────────────────────
// Avoids pulling in serde_json for the server binary.

fn json_str_field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!(r#""{key}""#);
    let start  = json.find(&needle)? + needle.len();
    let rest   = json[start..].trim_start();
    let rest   = rest.strip_prefix(':')?.trim_start();
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end   = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}

fn json_f64_field(json: &str, key: &str) -> Option<f64> {
    let needle = format!(r#""{key}""#);
    let start  = json.find(&needle)? + needle.len();
    let rest   = json[start..].trim_start();
    let rest   = rest.strip_prefix(':')?.trim_start();
    let end    = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = Arc::new(Args::parse());
    let n_pixels = (args.width * args.height) as usize;

    let state = Arc::new(Mutex::new(State {
        pixels:      vec![127u8; n_pixels],
        latest_jpeg: None,
        active:      None,
        stop_signal: false,
    }));

    // ── Encoder thread ────────────────────────────────────────────────────────
    // Snapshots the pixel buffer at `fps` intervals and JPEG-encodes it.
    {
        let state  = Arc::clone(&state);
        let args_c = Arc::clone(&args);
        thread::spawn(move || {
            let frame_interval = Duration::from_nanos(1_000_000_000 / args_c.fps);
            let mut snapshot   = vec![0u8; n_pixels];
            loop {
                thread::sleep(frame_interval);
                {
                    let st = state.lock().unwrap();
                    snapshot.copy_from_slice(&st.pixels);
                }
                let jpeg = encode_jpeg(&snapshot, args_c.width, args_c.height, args_c.quality);
                state.lock().unwrap().latest_jpeg = Some(jpeg);
            }
        });
    }

    // ── HTTP server ───────────────────────────────────────────────────────────
    let listener = TcpListener::bind(&args.bind)
        .unwrap_or_else(|e| panic!("Failed to bind {}: {e}", args.bind));

    println!("[playback] Listening on http://{}", args.bind);
    println!("[playback] MJPEG stream: http://{}/", args.bind);
    println!("[playback] POST /play   {{ \"file\": \"/path/to/file.raw\", \"speed\": 1.0 }}");
    println!("[playback] POST /stop");
    println!("[playback] GET  /status");

    for tcp_stream in listener.incoming().flatten() {
        let state  = Arc::clone(&state);
        let args_c = Arc::clone(&args);
        thread::spawn(move || handle_client(tcp_stream, state, args_c));
    }
}