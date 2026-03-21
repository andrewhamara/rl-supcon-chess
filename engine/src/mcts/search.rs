use cozy_chess::{Board, Move};
use rand::Rng;

use crate::board;
use crate::mcts::node::Node;
use crate::nn::inference::NnEvaluator;
use crate::policy;

/// MCTS search configuration.
#[derive(Clone)]
pub struct MctsConfig {
    /// Exploration constant for PUCT.
    pub c_puct: f32,
    /// Number of simulations per search.
    pub num_simulations: u32,
    /// Dirichlet noise alpha for root exploration.
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight (epsilon).
    pub dirichlet_epsilon: f32,
    /// Temperature for move selection.
    pub temperature: f32,
    /// Move number threshold for temperature annealing (tau -> 0 after this).
    pub temperature_threshold: u32,
    /// Batch size for leaf evaluation.
    pub eval_batch_size: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 2.5,
            num_simulations: 800,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
            temperature_threshold: 30,
            eval_batch_size: 8,
        }
    }
}

/// MCTS search engine.
pub struct MctsSearch {
    pub config: MctsConfig,
}

/// Neural network evaluation result for a single position.
pub struct NnOutput {
    /// Policy logits over all 1858 moves.
    pub policy: Vec<f32>,
    /// Scalar value prediction in [-1, 1].
    pub value: f32,
}

impl MctsSearch {
    pub fn new(config: MctsConfig) -> Self {
        MctsSearch { config }
    }

    /// Run MCTS from the given position and return the root node.
    pub fn search(&self, board: &Board, evaluator: &dyn NnEvaluator) -> Node {
        let mut root = Node::root();
        self.expand(&mut root, board, evaluator);
        self.add_dirichlet_noise(&mut root);

        // Collect leaves in batches for efficient NN evaluation
        for _ in 0..self.config.num_simulations {
            let mut path = Vec::new();
            let mut current_board = board.clone();

            // Selection: walk down the tree
            let leaf_value = self.select_and_expand(&mut root, &mut current_board, &mut path, evaluator);

            // Backpropagation: update all nodes on the path
            self.backpropagate(&path, leaf_value);
        }

        root
    }

    /// Select a leaf node, expand it, and return its value.
    /// `path` accumulates references to nodes along the path for backpropagation.
    fn select_and_expand(
        &self,
        root: &mut Node,
        board: &mut Board,
        path: &mut Vec<*const Node>,
        evaluator: &dyn NnEvaluator,
    ) -> f32 {
        let mut current = root as *mut Node;

        loop {
            let node = unsafe { &mut *current };

            // Terminal node check
            if board::is_game_over(board) {
                let result = board::game_result(board);
                // Result is from perspective of side to move, but we need
                // it from perspective of the parent (who made the last move)
                return -result;
            }

            if !node.is_expanded() {
                // Expand and evaluate
                self.expand(node, board, evaluator);
                let features = board::encode_board(board);
                let output = evaluator.evaluate(&features);
                // Value is from side to move's perspective; negate for parent
                return -output.value;
            }

            // Select best child by PUCT
            let parent_visits = node.visit_count();
            let children = node.children.as_ref().unwrap();

            let best_idx = children
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.puct_score(parent_visits, self.config.c_puct)
                        .partial_cmp(&b.puct_score(parent_visits, self.config.c_puct))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap();

            let child = &children[best_idx];
            child.add_virtual_loss();
            path.push(child as *const Node);

            // Apply the move
            board.play_unchecked(child.mv);

            // Move to child (we need a mutable pointer for potential expansion)
            current = &children[best_idx] as *const Node as *mut Node;
        }
    }

    /// Expand a node by generating all legal moves and evaluating with the NN.
    fn expand(&self, node: &mut Node, board: &Board, evaluator: &dyn NnEvaluator) {
        if node.expanded.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let moves = board::legal_moves(board);
        if moves.is_empty() {
            node.expanded.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        // Get policy from NN
        let features = board::encode_board(board);
        let output = evaluator.evaluate(&features);

        let perspective = board.side_to_move();

        // Extract priors for legal moves and softmax
        let move_logits: Vec<(Move, f32)> = moves
            .iter()
            .map(|&mv| {
                let idx = policy::encode_move(mv, perspective);
                let logit = if idx < output.policy.len() {
                    output.policy[idx]
                } else {
                    -10.0 // Very unlikely for out-of-range
                };
                (mv, logit)
            })
            .collect();

        // Softmax over legal moves
        let max_logit = move_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = move_logits.iter().map(|(_, l)| (l - max_logit).exp()).sum();

        let children: Vec<Node> = move_logits
            .iter()
            .map(|(mv, logit)| {
                let prior = (logit - max_logit).exp() / exp_sum;
                Node::new(*mv, prior)
            })
            .collect();

        node.children = Some(children.into_boxed_slice());
        node.expanded.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Add Dirichlet noise to the root node's children for exploration.
    fn add_dirichlet_noise(&self, root: &mut Node) {
        let children = match root.children.as_mut() {
            Some(c) => c,
            None => return,
        };

        let alpha = self.config.dirichlet_alpha;
        let epsilon = self.config.dirichlet_epsilon;
        let n = children.len();

        if n == 0 {
            return;
        }

        // Sample from Dirichlet distribution (using Gamma distribution)
        let mut rng = rand::thread_rng();
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                // Gamma(alpha, 1) sampling using Marsaglia and Tsang's method
                gamma_sample(&mut rng, alpha)
            })
            .collect();
        let noise_sum: f32 = noise.iter().sum();

        for (child, &n_i) in children.iter_mut().zip(noise.iter()) {
            let noise_val = n_i / noise_sum;
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise_val;
        }
    }

    /// Backpropagate a value up the path.
    fn backpropagate(&self, path: &[*const Node], mut value: f32) {
        for &node_ptr in path.iter().rev() {
            let node = unsafe { &*node_ptr };
            node.update(value);
            value = -value; // Flip perspective at each level
        }
    }

    /// Select a move from the root based on visit counts.
    pub fn select_move(&self, root: &Node, move_number: u32) -> Move {
        let temp = if move_number < self.config.temperature_threshold {
            self.config.temperature
        } else {
            0.0 // Greedy after threshold
        };

        let distribution = root.visit_distribution(temp);

        if temp < 1e-6 {
            // Greedy
            distribution
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0
        } else {
            // Sample proportional to visit count^(1/temp)
            let mut rng = rand::thread_rng();
            let r: f32 = rng.gen();
            let mut cumulative = 0.0;
            for (mv, prob) in &distribution {
                cumulative += prob;
                if r <= cumulative {
                    return *mv;
                }
            }
            distribution.last().unwrap().0
        }
    }
}

/// Sample from Gamma(alpha, 1) distribution.
fn gamma_sample(rng: &mut impl Rng, alpha: f32) -> f32 {
    if alpha < 1.0 {
        // Use Ahrens-Dieter method for alpha < 1
        let u: f32 = rng.gen();
        gamma_sample(rng, alpha + 1.0) * u.powf(1.0 / alpha)
    } else {
        // Marsaglia and Tsang's method
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x: f32 = {
                // Standard normal via Box-Muller
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            };
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u: f32 = rng.gen();
            if u < 1.0 - 0.0331 * x.powi(4)
                || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln())
            {
                return d * v;
            }
        }
    }
}
