use cozy_chess::Board;
use rayon::prelude::*;

use crate::board;
use crate::mcts::search::{MctsConfig, MctsSearch};
use crate::nn::inference::NnEvaluator;
use crate::policy;

/// A single training example from self-play.
#[derive(Clone)]
pub struct TrainingPosition {
    /// Board features [21*64 = 1344 f32 values].
    pub features: Vec<f32>,
    /// Policy target: sparse (move_index, visit_fraction) pairs.
    pub policy_target: Vec<(u16, f32)>,
    /// Value target: +1 (win), -1 (loss), 0 (draw) from side to move's perspective.
    pub value_target: f32,
}

/// Result of a self-play game.
pub struct GameResult {
    pub positions: Vec<TrainingPosition>,
    pub outcome: f32, // +1 white wins, -1 black wins, 0 draw
    pub num_moves: u32,
}

/// Generate a single self-play game.
pub fn play_game(config: &MctsConfig, evaluator: &dyn NnEvaluator) -> GameResult {
    let mut board = Board::default();
    let mut positions: Vec<(Vec<f32>, Vec<(u16, f32)>, cozy_chess::Color)> = Vec::new();
    let mut move_number = 0u32;

    loop {
        if board::is_game_over(&board) {
            break;
        }

        // Limit game length
        if move_number >= 512 {
            break;
        }

        let search = MctsSearch::new(config.clone());
        let root = search.search(&board, evaluator);

        // Record position
        let features = board::encode_board(&board);
        let perspective = board.side_to_move();

        // Get visit distribution as policy target
        let temp = if move_number < config.temperature_threshold {
            config.temperature
        } else {
            0.01 // Nearly greedy but not exactly 0 for numerical stability
        };
        let visit_dist = root.visit_distribution(temp);
        let policy_target: Vec<(u16, f32)> = visit_dist
            .iter()
            .filter(|(_, p)| *p > 1e-6)
            .map(|(mv, p)| {
                let idx = policy::encode_move(*mv, perspective);
                (idx as u16, *p)
            })
            .collect();

        positions.push((features.to_vec(), policy_target, perspective));

        // Select and play move
        let selected = search.select_move(&root, move_number);
        board.play_unchecked(selected);
        move_number += 1;
    }

    // Determine game outcome
    let outcome = if board::is_game_over(&board) {
        let result = board::game_result(&board);
        // game_result returns from side to move's perspective
        // We need from White's perspective for consistent labeling
        if board.side_to_move() == cozy_chess::Color::White {
            result
        } else {
            -result
        }
    } else {
        0.0 // Draw by move limit
    };

    // Assign value targets: game outcome from each position's side to move perspective
    let training_positions: Vec<TrainingPosition> = positions
        .into_iter()
        .map(|(features, policy_target, color)| {
            let value_target = if color == cozy_chess::Color::White {
                outcome
            } else {
                -outcome
            };
            TrainingPosition {
                features,
                policy_target,
                value_target,
            }
        })
        .collect();

    GameResult {
        num_moves: move_number,
        outcome,
        positions: training_positions,
    }
}

/// Generate multiple self-play games in parallel across all CPU cores.
pub fn generate_games(
    num_games: usize,
    config: &MctsConfig,
    evaluator: &dyn NnEvaluator,
) -> Vec<GameResult> {
    (0..num_games)
        .into_par_iter()
        .map(|_| play_game(config, evaluator))
        .collect()
}

/// Serialize training positions to a binary format for Python consumption.
pub fn serialize_positions(positions: &[TrainingPosition]) -> Vec<u8> {
    let mut data = Vec::new();

    // Header: number of positions
    let num_pos = positions.len() as u32;
    data.extend_from_slice(&num_pos.to_le_bytes());

    for pos in positions {
        // Features: 1344 f32 values
        for &f in &pos.features {
            data.extend_from_slice(&f.to_le_bytes());
        }

        // Policy target: num_moves, then (index, probability) pairs
        let num_moves = pos.policy_target.len() as u16;
        data.extend_from_slice(&num_moves.to_le_bytes());
        for &(idx, prob) in &pos.policy_target {
            data.extend_from_slice(&idx.to_le_bytes());
            data.extend_from_slice(&prob.to_le_bytes());
        }

        // Value target
        data.extend_from_slice(&pos.value_target.to_le_bytes());
    }

    data
}
