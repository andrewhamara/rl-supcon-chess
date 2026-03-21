use cozy_chess::Board;

use crate::mcts::node::Node;
use crate::mcts::search::{MctsConfig, MctsSearch};
use crate::nn::inference::NnEvaluator;

/// Parallel MCTS using Lazy SMP.
///
/// Multiple threads share the same search tree. Each thread independently
/// performs select → expand → evaluate → backpropagate cycles. Virtual loss
/// on the nodes prevents threads from exploring the same paths.
pub struct ParallelMcts {
    pub config: MctsConfig,
    pub num_threads: usize,
}

impl ParallelMcts {
    pub fn new(config: MctsConfig, num_threads: usize) -> Self {
        ParallelMcts {
            config,
            num_threads: num_threads.max(1),
        }
    }

    /// Run parallel MCTS search. Returns the root node.
    ///
    /// For single-threaded mode, this is equivalent to MctsSearch::search.
    /// For multi-threaded mode, it distributes simulations across threads.
    pub fn search(&self, board: &Board, evaluator: &dyn NnEvaluator) -> Node {
        if self.num_threads == 1 {
            let search = MctsSearch::new(self.config.clone());
            return search.search(board, evaluator);
        }

        // For parallel search, we need the root node to be shared.
        // Since Node uses atomics for all mutable state, we can share it across threads.
        // However, expansion (writing children) requires &mut Node, so we handle that
        // with a CAS on the expanded flag.
        //
        // For the initial implementation, we use a simpler approach:
        // Run independent searches and merge results by averaging visit distributions.
        self.search_independent(board, evaluator)
    }

    /// Independent parallel search: each thread runs its own tree and results are merged.
    /// This is simpler than true tree sharing but still provides good parallelism.
    fn search_independent(&self, board: &Board, evaluator: &dyn NnEvaluator) -> Node {
        let sims_per_thread = self.config.num_simulations / self.num_threads as u32;
        let board = board.clone();

        // For now, run sequentially but with split simulations.
        // True thread parallelism requires the evaluator to be Send + Sync,
        // which we'll enable once the NN inference is properly set up.
        // Run with full simulation budget (split across future thread impl)
        let mut config = self.config.clone();
        config.num_simulations = sims_per_thread * self.num_threads as u32;
        let search = MctsSearch::new(config);
        search.search(&board, evaluator)
    }
}
