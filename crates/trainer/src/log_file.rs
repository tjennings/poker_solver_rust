//! File-based logging for postflop solver output.
//!
//! When the TUI is active, stdout/stderr are owned by the terminal UI, so
//! solver diagnostic messages would be lost. This module captures all solver
//! output to a timestamped log file under `./logs/`, and conditionally mirrors
//! to stdout/stderr when the TUI is not running.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// Whether the TUI is currently rendering to the terminal.
/// When true, the macros suppress stdout/stderr output and only write to the
/// log file.
pub static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Global log file handle, initialized once per process.
static LOG_FILE: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();

/// Initialize the log file for this session.
///
/// Creates `./logs/` if it doesn't exist and opens a timestamped log file.
/// Safe to call multiple times — only the first call has effect.
pub fn init_log_file() {
    LOG_FILE.get_or_init(|| {
        let _ = fs::create_dir_all("logs");
        let timestamp = format_timestamp(SystemTime::now());
        let path = format!("logs/postflop_solve_{timestamp}.log");
        // If we can't open the log file, create a writer to /dev/null so
        // callers never need to handle the failure.
        let file = File::create(&path).unwrap_or_else(|e| {
            eprintln!("Warning: could not create log file {path}: {e}");
            File::create("/dev/null").expect("failed to open /dev/null")
        });
        Mutex::new(BufWriter::new(file))
    });
}

/// Write a pre-formatted line to the log file (if initialized).
#[doc(hidden)]
pub fn write_to_log(line: &str) {
    if let Some(lock) = LOG_FILE.get()
        && let Ok(mut writer) = lock.lock()
    {
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.write_all(b"\n");
        let _ = writer.flush();
    }
}

/// Format a `SystemTime` as `YYYYMMDD_HHMMSS` without external dependencies.
fn format_timestamp(time: SystemTime) -> String {
    let secs = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Civil time from unix timestamp (UTC).
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;

    // Days since epoch to (year, month, day) — civil calendar algorithm.
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}{month:02}{day:02}_{hour:02}{minute:02}{second:02}")
}

/// Convert days since 1970-01-01 to (year, month, day).
///
/// Uses the algorithm from Howard Hinnant's `chrono`-compatible date library.
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Log a diagnostic message (stderr-style).
///
/// Always writes to the log file. Also writes to stderr when the TUI is not
/// active.
#[macro_export]
macro_rules! solver_log {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::log_file::write_to_log(&msg);
        if !$crate::log_file::TUI_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("{}", msg);
        }
    }};
}

/// Log a result/output message (stdout-style).
///
/// Always writes to the log file. Also writes to stdout when the TUI is not
/// active.
#[macro_export]
macro_rules! solver_print {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        $crate::log_file::write_to_log(&msg);
        if !$crate::log_file::TUI_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            println!("{}", msg);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_format_is_valid() {
        let ts = format_timestamp(SystemTime::UNIX_EPOCH);
        assert_eq!(ts, "19700101_000000");
    }

    #[test]
    fn days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2024-01-01 is day 19723 since epoch
        let (y, m, d) = days_to_ymd(19723);
        assert_eq!((y, m, d), (2024, 1, 1));
    }

    #[test]
    fn tui_active_defaults_to_false() {
        // Ensure the default is false so macros print to terminal by default.
        assert!(!TUI_ACTIVE.load(std::sync::atomic::Ordering::Relaxed));
    }
}
