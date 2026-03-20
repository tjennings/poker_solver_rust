//! Binary format for storing k-means centroids per street.
//!
//! Used by the clustering pipeline to persist centroid vectors alongside
//! bucket files, enabling quality diagnostics and warm-start re-clustering.
//!
//! ## Wire format (little-endian)
//!
//! | Offset | Size | Field |
//! |-|-|-|
//! | 0 | 4 | Magic `CEN1` |
//! | 4 | 1 | Street (0-3) |
//! | 5 | 2 | K (u16, number of centroids) |
//! | 7 | 2 | dim (u16, dimension per centroid) |
//! | 9 | K*dim*8 | f64 centroid values |

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::Street;

const MAGIC: [u8; 4] = *b"CEN1";

/// Header size in bytes: 4 (magic) + 1 (street) + 2 (K) + 2 (dim) = 9
const HEADER_SIZE: usize = 9;

/// A centroid file holding k-means centroid vectors for a single street.
#[derive(Debug)]
pub struct CentroidFile {
    street: Street,
    centroids: Vec<Vec<f64>>,
}

impl CentroidFile {
    /// Create a new centroid file from a street and centroid vectors.
    #[must_use]
    pub fn new(street: Street, centroids: Vec<Vec<f64>>) -> Self {
        Self { street, centroids }
    }

    /// Returns the street this centroid file belongs to.
    #[must_use]
    pub fn street(&self) -> Street {
        self.street
    }

    /// Returns the centroid vectors.
    #[must_use]
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Save the centroid file to disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if file creation or writing fails.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Write the centroid file to an arbitrary writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing to the underlying writer fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&MAGIC)?;
        writer.write_all(&[self.street as u8])?;

        #[allow(clippy::cast_possible_truncation)]
        let k = self.centroids.len() as u16;
        writer.write_all(&k.to_le_bytes())?;

        #[allow(clippy::cast_possible_truncation)]
        let dim = self.centroids.first().map_or(0_u16, |c| c.len() as u16);
        writer.write_all(&dim.to_le_bytes())?;

        for centroid in &self.centroids {
            for &val in centroid {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load a centroid file from disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or contains invalid data.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    /// Read a centroid file from an arbitrary reader, validating header fields.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if:
    /// - The magic bytes do not match `CEN1`.
    /// - The street value is out of range.
    /// - Reading from the underlying reader fails.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header_buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_buf)?;

        // Validate magic bytes.
        if header_buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bad magic: expected {:?}, got {:?}",
                    MAGIC,
                    &header_buf[0..4]
                ),
            ));
        }

        let street = Street::from_u8(header_buf[4]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid street value: {}", header_buf[4]),
            )
        })?;

        let k = u16::from_le_bytes([header_buf[5], header_buf[6]]);
        let dim = u16::from_le_bytes([header_buf[7], header_buf[8]]);

        let total = k as usize * dim as usize;
        let mut data_buf = vec![0u8; total * 8];
        reader.read_exact(&mut data_buf)?;

        // Parse flat f64 values, then split into K centroid vectors of length dim.
        let flat: Vec<f64> = data_buf
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        let centroids: Vec<Vec<f64>> = flat
            .chunks_exact(dim as usize)
            .map(<[f64]>::to_vec)
            .collect();

        Ok(Self { street, centroids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn round_trip_centroid_file() {
        let dir = tempfile::tempdir().expect("tempdir failed");
        let path = dir.path().join("test.cen");

        let centroids = vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let original = CentroidFile::new(Street::Turn, centroids.clone());

        assert_eq!(original.street(), Street::Turn);
        assert_eq!(original.centroids().len(), 2);
        assert_eq!(original.centroids()[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(original.centroids()[1], vec![4.0, 5.0, 6.0]);

        original.save(&path).expect("save failed");
        let loaded = CentroidFile::load(&path).expect("load failed");

        assert_eq!(loaded.street(), Street::Turn);
        assert_eq!(loaded.centroids().len(), 2);
        assert_eq!(loaded.centroids()[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.centroids()[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn round_trip_scalar_centroids() {
        let dir = tempfile::tempdir().expect("tempdir failed");
        let path = dir.path().join("scalar.cen");

        let centroids = vec![
            vec![0.5_f64],
            vec![1.5],
            vec![2.5],
        ];
        let original = CentroidFile::new(Street::River, centroids.clone());

        original.save(&path).expect("save failed");
        let loaded = CentroidFile::load(&path).expect("load failed");

        assert_eq!(loaded.street(), Street::River);
        assert_eq!(loaded.centroids().len(), 3);
        assert_eq!(loaded.centroids()[0], vec![0.5]);
        assert_eq!(loaded.centroids()[1], vec![1.5]);
        assert_eq!(loaded.centroids()[2], vec![2.5]);
    }

    #[test]
    fn bad_magic_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir failed");
        let path = dir.path().join("bad.cen");

        // Write garbage bytes
        std::fs::write(&path, b"JUNK_garbage_data_here").expect("write failed");

        let err = CentroidFile::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("bad magic"),
            "unexpected error message: {err}"
        );
    }
}
