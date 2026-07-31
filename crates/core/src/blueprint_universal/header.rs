//! Fixed little-endian binary header for universal blueprint payload files.
//!
//! ## Wire format (little-endian, 48 bytes)
//!
//! | Offset | Size | Field              | Description                           |
//! |--------|------|--------------------|---------------------------------------|
//! | 0      | 8    | `magic`            | File-specific ASCII magic             |
//! | 8      | 2    | `format_version`   | Writer format version (starts at 1)   |
//! | 10     | 2    | `header_version`   | Header layout version (starts at 1)   |
//! | 12     | 1    | `endianness`       | 1 = little-endian                     |
//! | 13     | 7    | `reserved`         | Must be zero                          |
//! | 20     | 4    | `header_len`       | Total header length in bytes (48)     |
//! | 24     | 8    | `record_count`     | Number of records in payload          |
//! | 32     | 8    | `payload_len`      | Payload byte length (after header)    |
//! | 40     | 8    | `payload_crc64`    | CRC-64/XZ of the payload bytes        |

use super::error::FormatError;
use std::io::{Read, Write};

/// Total size of the binary header in bytes.
pub const HEADER_SIZE: usize = 48;

/// Endianness marker: 1 = little-endian (the only supported value).
const ENDIANNESS_LE: u8 = 1;

/// Current format version written by this implementation.
pub const CURRENT_FORMAT_VERSION: u16 = 1;

/// Current header layout version.
pub const CURRENT_HEADER_VERSION: u16 = 1;

/// Magic bytes for `strategy.rows.bin`.
pub const MAGIC_ROWS: [u8; 8] = *b"BPROW001";

/// Magic bytes for `strategy.actions.bin`.
pub const MAGIC_ACTIONS: [u8; 8] = *b"BPACT001";

/// Magic bytes for `strategy.probs.f32.bin`.
pub const MAGIC_PROBS: [u8; 8] = *b"BPPRO001";

/// Magic bytes for `strategy.semantic.bin`.
pub const MAGIC_SEMANTIC: [u8; 8] = *b"BPSEM001";

/// Parsed binary header for a universal blueprint payload file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryHeader {
    pub magic: [u8; 8],
    pub format_version: u16,
    pub header_version: u16,
    pub endianness: u8,
    pub header_len: u32,
    pub record_count: u64,
    pub payload_len: u64,
    pub payload_crc64: u64,
}

impl BinaryHeader {
    /// Create a new header with the given parameters.
    #[must_use]
    pub fn new(magic: [u8; 8], record_count: u64, payload_len: u64, payload_crc64: u64) -> Self {
        Self {
            magic,
            format_version: CURRENT_FORMAT_VERSION,
            header_version: CURRENT_HEADER_VERSION,
            endianness: ENDIANNESS_LE,
            #[allow(clippy::cast_possible_truncation)]
            header_len: HEADER_SIZE as u32,
            record_count,
            payload_len,
            payload_crc64,
        }
    }

    /// Serialize this header to a writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), FormatError> {
        w.write_all(&self.magic)?;
        w.write_all(&self.format_version.to_le_bytes())?;
        w.write_all(&self.header_version.to_le_bytes())?;
        w.write_all(&[self.endianness])?;
        w.write_all(&[0u8; 7])?; // reserved
        w.write_all(&self.header_len.to_le_bytes())?;
        w.write_all(&self.record_count.to_le_bytes())?;
        w.write_all(&self.payload_len.to_le_bytes())?;
        w.write_all(&self.payload_crc64.to_le_bytes())?;
        Ok(())
    }

    /// Deserialize a header from a reader, validating structural fields.
    ///
    /// # Errors
    ///
    /// Returns `FormatError::Truncated` if the reader has fewer than
    /// [`HEADER_SIZE`] bytes, or `FormatError::BadMagic` /
    /// `FormatError::UnsupportedFormatVersion` on validation failure.
    pub fn read_from<R: Read>(
        r: &mut R,
        expected_magic: [u8; 8],
        file_name: &str,
    ) -> Result<Self, FormatError> {
        let mut buf = [0u8; HEADER_SIZE];
        read_exact_or_truncated(r, &mut buf, file_name)?;
        parse_header(&buf, expected_magic, file_name)
    }
}

/// Read exactly `buf.len()` bytes, returning `Truncated` on short read.
fn read_exact_or_truncated<R: Read>(
    r: &mut R,
    buf: &mut [u8],
    file_name: &str,
) -> Result<(), FormatError> {
    let expected = buf.len();
    let mut total = 0;
    while total < expected {
        match r.read(&mut buf[total..]) {
            Ok(0) => {
                return Err(FormatError::Truncated {
                    file: file_name.to_string(),
                    expected,
                    actual: total,
                });
            }
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FormatError::Io(e)),
        }
    }
    Ok(())
}

/// Parse a header from a 48-byte buffer.
fn parse_header(
    buf: &[u8; HEADER_SIZE],
    expected_magic: [u8; 8],
    file_name: &str,
) -> Result<BinaryHeader, FormatError> {
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&buf[0..8]);

    if magic != expected_magic {
        return Err(FormatError::BadMagic {
            file: file_name.to_string(),
            expected: String::from_utf8_lossy(&expected_magic).to_string(),
            actual: String::from_utf8_lossy(&magic).to_string(),
        });
    }

    let format_version = u16::from_le_bytes([buf[8], buf[9]]);
    if format_version > CURRENT_FORMAT_VERSION {
        return Err(FormatError::UnsupportedFormatVersion {
            file: file_name.to_string(),
            version: format_version,
            max_supported: CURRENT_FORMAT_VERSION,
        });
    }

    let header_version = u16::from_le_bytes([buf[10], buf[11]]);
    let endianness = buf[12];
    // reserved: buf[13..20] -- ignored on read
    let header_len = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let record_count = u64::from_le_bytes(buf[24..32].try_into().unwrap());
    let payload_len = u64::from_le_bytes(buf[32..40].try_into().unwrap());
    let payload_crc64 = u64::from_le_bytes(buf[40..48].try_into().unwrap());

    Ok(BinaryHeader {
        magic,
        format_version,
        header_version,
        endianness,
        header_len,
        record_count,
        payload_len,
        payload_crc64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_round_trips() {
        let header = BinaryHeader::new(MAGIC_ROWS, 42, 1024, 0xDEAD_BEEF);
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let loaded = BinaryHeader::read_from(&mut cursor, MAGIC_ROWS, "test.bin").unwrap();
        assert_eq!(loaded, header);
    }

    #[test]
    fn rejects_wrong_magic() {
        let header = BinaryHeader::new(MAGIC_ROWS, 0, 0, 0);
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = BinaryHeader::read_from(&mut cursor, MAGIC_ACTIONS, "test.bin").unwrap_err();
        assert!(matches!(err, FormatError::BadMagic { .. }));
    }

    #[test]
    fn rejects_future_version() {
        let mut header = BinaryHeader::new(MAGIC_ROWS, 0, 0, 0);
        header.format_version = 99;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = BinaryHeader::read_from(&mut cursor, MAGIC_ROWS, "test.bin").unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFormatVersion { .. }));
    }

    #[test]
    fn rejects_truncated_buffer() {
        let buf = vec![0u8; 10]; // too short
        let mut cursor = Cursor::new(&buf);
        let err = BinaryHeader::read_from(&mut cursor, MAGIC_ROWS, "test.bin").unwrap_err();
        assert!(matches!(err, FormatError::Truncated { .. }));
    }
}
