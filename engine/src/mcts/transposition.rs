use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free transposition table for MCTS.
///
/// Stores position evaluations to avoid re-evaluating the same position.
/// Uses Zobrist hash keys for position identification.
///
/// Entry layout (16 bytes):
/// - key: u64 (Zobrist hash XOR'd with data for verification)
/// - data: u64 (packed: value i16, depth u8, flag u8, best_move u16, padding u16)
#[repr(align(64))] // Cache-line aligned
pub struct TranspositionTable {
    entries: Box<[TtEntry]>,
    mask: usize,
}

#[repr(align(16))]
struct TtEntry {
    key: AtomicU64,
    data: AtomicU64,
}

/// Flags for TT entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TtFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

/// Stored TT entry data.
#[derive(Debug, Clone, Copy)]
pub struct TtData {
    pub value: f32,
    pub depth: u8,
    pub flag: TtFlag,
    pub best_move_idx: u16,
}

impl TranspositionTable {
    /// Create a new transposition table with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TtEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round down to power of 2 for fast masking
        let num_entries = num_entries.next_power_of_two() >> 1;
        let num_entries = num_entries.max(1024);

        let entries: Vec<TtEntry> = (0..num_entries)
            .map(|_| TtEntry {
                key: AtomicU64::new(0),
                data: AtomicU64::new(0),
            })
            .collect();

        TranspositionTable {
            entries: entries.into_boxed_slice(),
            mask: num_entries - 1,
        }
    }

    /// Probe the table for a position.
    pub fn probe(&self, hash: u64) -> Option<TtData> {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let stored_key = entry.key.load(Ordering::Relaxed);
        let stored_data = entry.data.load(Ordering::Relaxed);

        // Verify key (XOR scheme: stored_key = hash ^ data)
        if stored_key ^ stored_data != hash {
            return None;
        }

        Some(unpack_data(stored_data))
    }

    /// Store a position evaluation in the table.
    /// Uses replace-by-depth scheme.
    pub fn store(&self, hash: u64, data: TtData) {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let packed = pack_data(&data);

        // Replace-by-depth: only overwrite if new depth >= existing depth
        let existing_data = entry.data.load(Ordering::Relaxed);
        if existing_data != 0 {
            let existing = unpack_data(existing_data);
            if data.depth < existing.depth {
                return; // Don't overwrite deeper entries
            }
        }

        // XOR scheme for key verification
        entry.data.store(packed, Ordering::Relaxed);
        entry.key.store(hash ^ packed, Ordering::Relaxed);
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for entry in self.entries.iter() {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }

    /// Get the number of entries in use (for stats).
    pub fn usage_permille(&self) -> u32 {
        let sample_size = 1000.min(self.entries.len());
        let used = self.entries[..sample_size]
            .iter()
            .filter(|e| e.data.load(Ordering::Relaxed) != 0)
            .count();
        (used * 1000 / sample_size) as u32
    }
}

fn pack_data(data: &TtData) -> u64 {
    let value_bits = (data.value * 10000.0) as i16 as u16;
    let flag_bits = data.flag as u8;

    (value_bits as u64)
        | ((data.depth as u64) << 16)
        | ((flag_bits as u64) << 24)
        | ((data.best_move_idx as u64) << 32)
}

fn unpack_data(packed: u64) -> TtData {
    let value_bits = packed as u16 as i16;
    let depth = (packed >> 16) as u8;
    let flag_bits = (packed >> 24) as u8;
    let best_move_idx = (packed >> 32) as u16;

    TtData {
        value: value_bits as f32 / 10000.0,
        depth,
        flag: match flag_bits {
            0 => TtFlag::Exact,
            1 => TtFlag::LowerBound,
            2 => TtFlag::UpperBound,
            _ => TtFlag::Exact,
        },
        best_move_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_probe() {
        let tt = TranspositionTable::new(1); // 1 MB
        let hash = 0x123456789ABCDEF0;
        let data = TtData {
            value: 0.75,
            depth: 10,
            flag: TtFlag::Exact,
            best_move_idx: 42,
        };

        tt.store(hash, data);
        let probed = tt.probe(hash).expect("Should find entry");
        assert!((probed.value - 0.75).abs() < 0.001);
        assert_eq!(probed.depth, 10);
        assert_eq!(probed.flag, TtFlag::Exact);
        assert_eq!(probed.best_move_idx, 42);
    }

    #[test]
    fn test_probe_miss() {
        let tt = TranspositionTable::new(1);
        assert!(tt.probe(0xDEADBEEF).is_none());
    }
}
