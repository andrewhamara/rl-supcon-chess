use cozy_chess::{Color, File, Move, Piece, Rank, Square};

/// AlphaZero-style move encoding for chess.
///
/// Each move is encoded as (from_square, move_type) where move_type is one of:
/// - 56 "queen moves": 7 distances x 8 directions (N, NE, E, SE, S, SW, W, NW)
/// - 8 knight moves: 8 possible L-shaped jumps
/// - 9 underpromotions: 3 directions (left-capture, forward, right-capture) x 3 pieces (N, B, R)
///
/// Total: 73 move types x 64 source squares = 4672 indices
/// But we use the Leela Chess Zero convention of 1858 unique moves instead,
/// which maps each legal move to a compact index.
///
/// For simplicity and compatibility, we use the flat 73x64 = 4672 encoding
/// but only the legal move indices matter during training.

pub const POLICY_SIZE: usize = 1858;

/// Queen move directions: (delta_rank, delta_file)
const QUEEN_DIRS: [(i8, i8); 8] = [
    (1, 0),   // N
    (1, 1),   // NE
    (0, 1),   // E
    (-1, 1),  // SE
    (-1, 0),  // S
    (-1, -1), // SW
    (0, -1),  // W
    (1, -1),  // NW
];

/// Knight move deltas: (delta_rank, delta_file)
const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
];

/// Underpromotion file deltas: left capture (-1), forward (0), right capture (+1)
const PROMO_FILE_DELTAS: [i8; 3] = [-1, 0, 1];

/// Underpromotion pieces (queen promotion is encoded as a queen move)
const UNDERPROMO_PIECES: [Piece; 3] = [Piece::Knight, Piece::Bishop, Piece::Rook];

/// Encode a move to a policy index (0..POLICY_SIZE-1).
/// The move is from the perspective of the side to move.
pub fn encode_move(mv: Move, perspective: Color) -> usize {
    let from = flip_square(mv.from, perspective);
    let to = flip_square(mv.to, perspective);

    let from_rank = from.rank() as i8;
    let from_file = from.file() as i8;
    let to_rank = to.rank() as i8;
    let to_file = to.file() as i8;

    let dr = to_rank - from_rank;
    let df = to_file - from_file;

    // Check for underpromotion
    if let Some(promo) = mv.promotion {
        if promo != Piece::Queen {
            return encode_underpromotion(from, df, promo);
        }
        // Queen promotion falls through to queen-move encoding
    }

    // Check for knight move
    if is_knight_delta(dr, df) {
        return encode_knight_move(from, dr, df);
    }

    // Queen-style move (includes queen promotions)
    encode_queen_move(from, dr, df)
}

/// Decode a policy index back to (from_square, to_square, promotion).
/// Returns None if the index is invalid.
pub fn decode_move(index: usize, perspective: Color) -> Option<Move> {
    if index >= POLICY_SIZE {
        return None;
    }

    // Look up in the move table
    let (from, to, promo) = MOVE_TABLE[index];
    let from = flip_square(from, perspective);
    let to = flip_square(to, perspective);
    Some(Move { from, to, promotion: promo })
}

fn flip_square(sq: Square, perspective: Color) -> Square {
    if perspective == Color::White {
        sq
    } else {
        let rank = 7 - sq.rank() as u8;
        let file = 7 - sq.file() as u8;
        Square::new(
            File::index(file as usize),
            Rank::index(rank as usize),
        )
    }
}

fn is_knight_delta(dr: i8, df: i8) -> bool {
    let adr = dr.abs();
    let adf = df.abs();
    (adr == 2 && adf == 1) || (adr == 1 && adf == 2)
}

fn encode_queen_move(from: Square, dr: i8, df: i8) -> usize {
    let dir = direction_index(dr, df);
    let dist = dr.abs().max(df.abs()) as usize - 1; // 0-indexed distance (0..6)
    let move_type = dir * 7 + dist; // 0..55
    let from_idx = from.rank() as usize * 8 + from.file() as usize;

    // Map to compact index using the precomputed table
    QUEEN_MOVE_OFFSETS[from_idx][move_type]
}

fn encode_knight_move(from: Square, dr: i8, df: i8) -> usize {
    let knight_idx = KNIGHT_DELTAS
        .iter()
        .position(|&(r, f)| r == dr && f == df)
        .unwrap();
    let from_idx = from.rank() as usize * 8 + from.file() as usize;
    KNIGHT_MOVE_OFFSETS[from_idx][knight_idx]
}

fn encode_underpromotion(from: Square, df: i8, piece: Piece) -> usize {
    let file_delta_idx = PROMO_FILE_DELTAS
        .iter()
        .position(|&d| d == df)
        .unwrap();
    let piece_idx = match piece {
        Piece::Knight => 0,
        Piece::Bishop => 1,
        Piece::Rook => 2,
        _ => unreachable!(),
    };
    let promo_type = file_delta_idx * 3 + piece_idx; // 0..8
    let from_file = from.file() as usize;
    UNDERPROMO_OFFSETS[from_file][promo_type]
}

fn direction_index(dr: i8, df: i8) -> usize {
    let norm_r = dr.signum();
    let norm_f = df.signum();
    match (norm_r, norm_f) {
        (1, 0) => 0,
        (1, 1) => 1,
        (0, 1) => 2,
        (-1, 1) => 3,
        (-1, 0) => 4,
        (-1, -1) => 5,
        (0, -1) => 6,
        (1, -1) => 7,
        _ => unreachable!("Invalid direction ({}, {})", dr, df),
    }
}

// ---- Precomputed tables ----
// These are generated at compile time to map (square, move_type) -> compact index.
// We use a simpler runtime approach: enumerate all legal source-dest pairs and assign indices.

use std::sync::LazyLock;

struct MoveTables {
    queen_offsets: [[usize; 56]; 64],   // [from_sq][dir*7+dist] -> index
    knight_offsets: [[usize; 8]; 64],    // [from_sq][knight_idx] -> index
    underpromo_offsets: [[usize; 9]; 8], // [from_file][type] -> index
    move_table: Vec<(Square, Square, Option<Piece>)>, // index -> (from, to, promo)
}

static TABLES: LazyLock<MoveTables> = LazyLock::new(build_tables);

static QUEEN_MOVE_OFFSETS: LazyLock<[[usize; 56]; 64]> =
    LazyLock::new(|| TABLES.queen_offsets);
static KNIGHT_MOVE_OFFSETS: LazyLock<[[usize; 8]; 64]> =
    LazyLock::new(|| TABLES.knight_offsets);
static UNDERPROMO_OFFSETS: LazyLock<[[usize; 9]; 8]> =
    LazyLock::new(|| TABLES.underpromo_offsets);
static MOVE_TABLE: LazyLock<Vec<(Square, Square, Option<Piece>)>> =
    LazyLock::new(|| TABLES.move_table.clone());

fn build_tables() -> MoveTables {
    let mut queen_offsets = [[usize::MAX; 56]; 64];
    let mut knight_offsets = [[usize::MAX; 8]; 64];
    let mut underpromo_offsets = [[usize::MAX; 9]; 8];
    let mut move_table: Vec<(Square, Square, Option<Piece>)> = Vec::new();
    let mut idx = 0;

    // Queen moves: for each square, each direction, each distance
    for from_rank in 0..8i8 {
        for from_file in 0..8i8 {
            let from_idx = (from_rank * 8 + from_file) as usize;
            let from_sq = Square::new(
                File::index(from_file as usize),
                Rank::index(from_rank as usize),
            );

            for (dir_idx, &(dr, df)) in QUEEN_DIRS.iter().enumerate() {
                for dist in 1..=7i8 {
                    let to_rank = from_rank + dr * dist;
                    let to_file = from_file + df * dist;
                    if to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7 {
                        break;
                    }
                    let to_sq = Square::new(
                        File::index(to_file as usize),
                        Rank::index(to_rank as usize),
                    );
                    let move_type = dir_idx * 7 + (dist as usize - 1);
                    queen_offsets[from_idx][move_type] = idx;

                    // Queen promotions: if pawn reaches last rank
                    if from_rank == 6 && to_rank == 7 && (dr != 0) {
                        move_table.push((from_sq, to_sq, Some(Piece::Queen)));
                    } else {
                        move_table.push((from_sq, to_sq, None));
                    }
                    idx += 1;
                }
            }
        }
    }

    // Knight moves
    for from_rank in 0..8i8 {
        for from_file in 0..8i8 {
            let from_idx = (from_rank * 8 + from_file) as usize;
            let from_sq = Square::new(
                File::index(from_file as usize),
                Rank::index(from_rank as usize),
            );

            for (knight_idx, &(dr, df)) in KNIGHT_DELTAS.iter().enumerate() {
                let to_rank = from_rank + dr;
                let to_file = from_file + df;
                if to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7 {
                    continue;
                }
                let to_sq = Square::new(
                    File::index(to_file as usize),
                    Rank::index(to_rank as usize),
                );
                knight_offsets[from_idx][knight_idx] = idx;
                move_table.push((from_sq, to_sq, None));
                idx += 1;
            }
        }
    }

    // Underpromotions: pawn on rank 6 promoting to N/B/R
    for from_file in 0..8i8 {
        let from_sq = Square::new(
            File::index(from_file as usize),
            Rank::index(6), // rank 7 (0-indexed 6) for white
        );

        for (fd_idx, &file_delta) in PROMO_FILE_DELTAS.iter().enumerate() {
            let to_file = from_file + file_delta;
            if to_file < 0 || to_file > 7 {
                continue;
            }
            let to_sq = Square::new(
                File::index(to_file as usize),
                Rank::index(7),
            );

            for (piece_idx, &piece) in UNDERPROMO_PIECES.iter().enumerate() {
                let promo_type = fd_idx * 3 + piece_idx;
                underpromo_offsets[from_file as usize][promo_type] = idx;
                move_table.push((from_sq, to_sq, Some(piece)));
                idx += 1;
            }
        }
    }

    assert_eq!(idx, move_table.len());
    // idx should be 1858 for standard chess

    MoveTables {
        queen_offsets,
        knight_offsets,
        underpromo_offsets,
        move_table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_size() {
        let tables = build_tables();
        // Verify the total number of moves matches POLICY_SIZE
        println!("Total encoded moves: {}", tables.move_table.len());
        assert_eq!(tables.move_table.len(), POLICY_SIZE,
            "Expected {} moves but got {}", POLICY_SIZE, tables.move_table.len());
    }

    #[test]
    fn test_encode_decode_e2e4() {
        let mv = Move {
            from: Square::new(File::E, Rank::Second),
            to: Square::new(File::E, Rank::Fourth),
            promotion: None,
        };
        let idx = encode_move(mv, Color::White);
        assert!(idx < POLICY_SIZE, "Index {} out of range", idx);
        let decoded = decode_move(idx, Color::White).unwrap();
        assert_eq!(decoded.from, mv.from);
        assert_eq!(decoded.to, mv.to);
        assert_eq!(decoded.promotion, mv.promotion);
    }

    #[test]
    fn test_encode_knight_move() {
        let mv = Move {
            from: Square::new(File::G, Rank::First),
            to: Square::new(File::F, Rank::Third),
            promotion: None,
        };
        let idx = encode_move(mv, Color::White);
        assert!(idx < POLICY_SIZE);
        let decoded = decode_move(idx, Color::White).unwrap();
        assert_eq!(decoded.from, mv.from);
        assert_eq!(decoded.to, mv.to);
    }
}
