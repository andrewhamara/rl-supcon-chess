use cozy_chess::{Board, Color, Move, Piece, Square};

/// Number of input feature planes for the neural network.
/// 6 piece types x 2 colors = 12 piece planes
/// + 1 side-to-move plane
/// + 4 castling rights planes (KQkq)
/// + 1 en passant plane
/// + 1 halfmove clock plane (normalized)
/// + 1 fullmove number plane (normalized)
/// + 1 has-legal-moves plane (all ones — padding)
/// = 21 planes
pub const NUM_INPUT_PLANES: usize = 21;
pub const BOARD_SIZE: usize = 8;
pub const INPUT_SIZE: usize = NUM_INPUT_PLANES * BOARD_SIZE * BOARD_SIZE;

/// Piece indices for feature planes (white pieces 0-5, black pieces 6-11).
const PIECE_PLANE: [usize; 6] = [0, 1, 2, 3, 4, 5]; // P, N, B, R, Q, K

/// Encode a board position into neural network input features.
/// Output shape: [21][8][8] flattened to [21*64] in NCHW order.
/// Features are from the perspective of the side to move.
pub fn encode_board(board: &Board) -> [f32; INPUT_SIZE] {
    let mut features = [0.0f32; INPUT_SIZE];
    let perspective = board.side_to_move();

    // Piece planes (0-11): 6 piece types x 2 colors
    for &color in &[Color::White, Color::Black] {
        for &piece in &[
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            let bb = board.colored_pieces(color, piece);
            let color_offset = if color == perspective { 0 } else { 6 };
            let plane = color_offset + PIECE_PLANE[piece as usize];

            for sq in bb {
                let (rank, file) = square_to_rf(sq, perspective);
                features[plane * 64 + rank * 8 + file] = 1.0;
            }
        }
    }

    // Plane 12: side to move (all 1s if white to move from perspective)
    if perspective == Color::White {
        for i in 0..64 {
            features[12 * 64 + i] = 1.0;
        }
    }

    // Planes 13-16: castling rights
    let rights = board.castle_rights(Color::White);
    if rights.short.is_some() {
        fill_plane(&mut features, 13, 1.0);
    }
    if rights.long.is_some() {
        fill_plane(&mut features, 14, 1.0);
    }
    let rights = board.castle_rights(Color::Black);
    if rights.short.is_some() {
        fill_plane(&mut features, 15, 1.0);
    }
    if rights.long.is_some() {
        fill_plane(&mut features, 16, 1.0);
    }

    // Plane 17: en passant square
    if let Some(ep_file) = board.en_passant() {
        let ep_rank = if board.side_to_move() == Color::White {
            5usize // rank 6 (0-indexed)
        } else {
            2usize // rank 3 (0-indexed)
        };
        let (r, f) = if perspective == Color::White {
            (ep_rank, ep_file as usize)
        } else {
            (7 - ep_rank, 7 - ep_file as usize)
        };
        features[17 * 64 + r * 8 + f] = 1.0;
    }

    // Plane 18: halfmove clock (normalized to [0, 1], clamped at 100)
    let halfmove = board.halfmove_clock().min(100) as f32 / 100.0;
    fill_plane(&mut features, 18, halfmove);

    // Plane 19: fullmove number (normalized, clamped at 200)
    let fullmove = (board.fullmove_number().min(200) as f32) / 200.0;
    fill_plane(&mut features, 19, fullmove);

    // Plane 20: all ones (constant plane)
    fill_plane(&mut features, 20, 1.0);

    features
}

/// Convert a square to (rank, file) from the given perspective.
/// If perspective is Black, the board is flipped.
fn square_to_rf(sq: Square, perspective: Color) -> (usize, usize) {
    let rank = sq.rank() as usize;
    let file = sq.file() as usize;
    if perspective == Color::White {
        (rank, file)
    } else {
        (7 - rank, 7 - file)
    }
}

/// Fill an entire plane with a constant value.
fn fill_plane(features: &mut [f32; INPUT_SIZE], plane: usize, value: f32) {
    let start = plane * 64;
    for i in start..start + 64 {
        features[i] = value;
    }
}

/// Get all legal moves for the current position.
pub fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    board.generate_moves(|mvs| {
        moves.extend(mvs);
        false
    });
    moves
}

/// Check if the game is over (checkmate, stalemate, or draw by rule).
pub fn is_game_over(board: &Board) -> bool {
    if board.halfmove_clock() >= 100 {
        return true; // 50-move rule
    }
    if has_insufficient_material(board) {
        return true;
    }
    legal_moves(board).is_empty()
}

/// Get game result from the perspective of the side to move.
/// Returns +1.0 for win, -1.0 for loss, 0.0 for draw.
/// Only valid when is_game_over() returns true.
pub fn game_result(board: &Board) -> f32 {
    let moves = legal_moves(board);
    if moves.is_empty() {
        if board.checkers().is_empty() {
            0.0 // stalemate
        } else {
            -1.0 // checkmate (side to move lost)
        }
    } else {
        0.0 // draw by rule
    }
}

/// Simple insufficient material check.
fn has_insufficient_material(board: &Board) -> bool {
    // If any pawns, rooks, or queens exist, material is sufficient
    if !(board.pieces(Piece::Pawn)
        | board.pieces(Piece::Rook)
        | board.pieces(Piece::Queen))
        .is_empty()
    {
        return false;
    }
    // K vs K
    let all = board.occupied();
    let piece_count = all.len();
    if piece_count <= 2 {
        return true;
    }
    // K+minor vs K
    if piece_count == 3 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_starting_position() {
        let board = Board::default();
        let features = encode_board(&board);
        assert_eq!(features.len(), INPUT_SIZE);

        // White pawns should be on rank 1 (index perspective of white)
        // Plane 0 = white pawns from white's perspective
        for file in 0..8 {
            assert_eq!(features[0 * 64 + 1 * 8 + file], 1.0, "White pawn at file {}", file);
        }
        // Black pawns on plane 6 (opponent pawns), rank 6 from white's perspective
        for file in 0..8 {
            assert_eq!(features[6 * 64 + 6 * 8 + file], 1.0, "Black pawn at file {}", file);
        }
    }

    #[test]
    fn test_legal_moves_starting() {
        let board = Board::default();
        let moves = legal_moves(&board);
        assert_eq!(moves.len(), 20); // 16 pawn + 4 knight moves
    }

    #[test]
    fn test_game_not_over_at_start() {
        let board = Board::default();
        assert!(!is_game_over(&board));
    }
}
