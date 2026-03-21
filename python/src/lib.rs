use pyo3::prelude::*;
use pyo3::types::PyBytes;

use chess_engine::mcts::search::{MctsConfig, MctsSearch};
use chess_engine::nn::inference::{NnEvaluator, RandomEvaluator};
use chess_engine::nn::weights::NetworkWeights;
use chess_engine::nn::simd::Int8Evaluator;
use chess_engine::selfplay;
use rayon::ThreadPoolBuilder;

/// Python bindings for the chess engine.
#[pymodule]
fn chess_engine_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(run_selfplay, m)?)?;
    m.add_function(wrap_pyfunction!(run_selfplay_random, m)?)?;
    m.add_function(wrap_pyfunction!(search_move, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    "0.1.0".to_string()
}

/// Load an evaluator from a weights file, preferring int8 quantized.
fn load_evaluator(weights_path: &str) -> Box<dyn NnEvaluator> {
    let path = std::path::Path::new(weights_path);
    if path.exists() {
        match NetworkWeights::load(path) {
            Ok(weights) => Box::new(Int8Evaluator::from_weights(&weights)),
            Err(_) => Box::new(RandomEvaluator),
        }
    } else {
        Box::new(RandomEvaluator)
    }
}

/// Run self-play games using the neural network weights from a file.
/// Uses int8 quantized inference for speed. Games run in parallel across `num_threads` cores.
#[pyfunction]
#[pyo3(signature = (weights_path, num_games=10, simulations=800, c_puct=2.5,
                     dirichlet_alpha=0.3, dirichlet_epsilon=0.25, temperature=1.0,
                     num_threads=0))]
fn run_selfplay(
    py: Python<'_>,
    weights_path: &str,
    num_games: usize,
    simulations: u32,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    temperature: f32,
    num_threads: usize,
) -> PyResult<Py<PyBytes>> {
    let config = MctsConfig {
        c_puct,
        num_simulations: simulations,
        dirichlet_alpha,
        dirichlet_epsilon,
        temperature,
        temperature_threshold: 30,
        eval_batch_size: 8,
    };

    let evaluator = load_evaluator(weights_path);

    let results = py.allow_threads(|| {
        // Build a rayon pool with explicit thread count (0 = use all cores)
        let threads = if num_threads > 0 { num_threads } else { num_cpus() };
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to create thread pool");
        pool.install(|| {
            selfplay::generate_games(num_games, &config, evaluator.as_ref())
        })
    });

    let all_positions: Vec<selfplay::TrainingPosition> = results
        .into_iter()
        .flat_map(|g| g.positions)
        .collect();

    let data = selfplay::serialize_positions(&all_positions);
    Ok(PyBytes::new_bound(py, &data).into())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Run self-play games using random evaluation (for testing).
#[pyfunction]
#[pyo3(signature = (num_games=10, simulations=100))]
fn run_selfplay_random(
    py: Python<'_>,
    num_games: usize,
    simulations: u32,
) -> PyResult<Py<PyBytes>> {
    let config = MctsConfig {
        num_simulations: simulations,
        ..MctsConfig::default()
    };

    let evaluator = RandomEvaluator;

    let results = py.allow_threads(|| {
        selfplay::generate_games(num_games, &config, &evaluator)
    });

    let all_positions: Vec<selfplay::TrainingPosition> = results
        .into_iter()
        .flat_map(|g| g.positions)
        .collect();

    let data = selfplay::serialize_positions(&all_positions);
    Ok(PyBytes::new_bound(py, &data).into())
}

/// Search for the best move from a FEN position.
/// Returns the best move in UCI format (e.g., "e2e4").
#[pyfunction]
#[pyo3(signature = (fen, weights_path="", simulations=200, c_puct=2.5))]
fn search_move(
    py: Python<'_>,
    fen: &str,
    weights_path: &str,
    simulations: u32,
    c_puct: f32,
) -> PyResult<String> {
    let board: cozy_chess::Board = fen.parse().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN: {}", e))
    })?;

    let config = MctsConfig {
        c_puct,
        num_simulations: simulations,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        temperature: 0.0, // Greedy for play
        temperature_threshold: 0,
        eval_batch_size: 8,
    };

    let evaluator = if weights_path.is_empty() {
        Box::new(RandomEvaluator) as Box<dyn NnEvaluator>
    } else {
        load_evaluator(weights_path)
    };

    let best_move = py.allow_threads(|| {
        let search = MctsSearch::new(config);
        let root = search.search(&board, evaluator.as_ref());
        search.select_move(&root, 999) // high move number = greedy
    });

    Ok(format!("{}{}{}", best_move.from, best_move.to,
        match best_move.promotion {
            Some(p) => match p {
                cozy_chess::Piece::Queen => "q",
                cozy_chess::Piece::Rook => "r",
                cozy_chess::Piece::Bishop => "b",
                cozy_chess::Piece::Knight => "n",
                _ => "",
            },
            None => "",
        }
    ))
}
