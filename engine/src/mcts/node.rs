use cozy_chess::Move;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};

/// A node in the MCTS search tree.
///
/// Uses atomic fields for lock-free Lazy SMP parallel search.
/// Values are stored from the perspective of the parent (the player who made the move).
pub struct Node {
    /// The move that led to this node from its parent.
    pub mv: Move,
    /// Prior probability from policy network P(s, a).
    pub prior: f32,
    /// Visit count N(s, a).
    pub visits: AtomicU32,
    /// Total value W(s, a), stored as fixed-point (value * FIXED_POINT_SCALE).
    /// Using i64 for enough precision and range.
    pub total_value: AtomicI64,
    /// Virtual loss counter for parallel search.
    pub virtual_loss: AtomicU32,
    /// Whether this node has been expanded (children generated).
    pub expanded: AtomicBool,
    /// Child nodes. None until expansion.
    pub children: Option<Box<[Node]>>,
}

const FIXED_POINT_SCALE: f64 = 1_000_000.0;

impl Node {
    /// Create a new unexpanded node.
    pub fn new(mv: Move, prior: f32) -> Self {
        Node {
            mv,
            prior,
            visits: AtomicU32::new(0),
            total_value: AtomicI64::new(0),
            virtual_loss: AtomicU32::new(0),
            expanded: AtomicBool::new(false),
            children: None,
        }
    }

    /// Create a root node (no move).
    pub fn root() -> Self {
        Node {
            mv: Move {
                from: cozy_chess::Square::A1,
                to: cozy_chess::Square::A1,
                promotion: None,
            },
            prior: 1.0,
            visits: AtomicU32::new(0),
            total_value: AtomicI64::new(0),
            virtual_loss: AtomicU32::new(0),
            expanded: AtomicBool::new(false),
            children: None,
        }
    }

    /// Get the mean action value Q(s, a).
    /// Accounts for virtual loss: treats each virtual loss as a loss (-1).
    pub fn q_value(&self) -> f32 {
        let visits = self.visits.load(Ordering::Relaxed);
        let vl = self.virtual_loss.load(Ordering::Relaxed);
        let total_visits = visits + vl;
        if total_visits == 0 {
            return 0.0;
        }
        let total = self.total_value.load(Ordering::Relaxed) as f64 / FIXED_POINT_SCALE;
        let vl_penalty = vl as f64; // Each virtual loss counts as -1
        ((total - vl_penalty) / total_visits as f64) as f32
    }

    /// Get visit count.
    pub fn visit_count(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    /// Add virtual loss (called during selection).
    pub fn add_virtual_loss(&self) {
        self.virtual_loss.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove virtual loss and update value (called during backpropagation).
    /// `value` is from the perspective of the player who made this move.
    pub fn update(&self, value: f32) {
        self.virtual_loss.fetch_sub(1, Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
        let scaled = (value as f64 * FIXED_POINT_SCALE) as i64;
        self.total_value.fetch_add(scaled, Ordering::Relaxed);
    }

    /// PUCT score for child selection.
    /// parent_visits: N(s) — total visits of the parent node.
    /// c_puct: exploration constant.
    pub fn puct_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let visits = self.visits.load(Ordering::Relaxed) + self.virtual_loss.load(Ordering::Relaxed);
        let q = self.q_value();
        let u = c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + visits as f32);
        q + u
    }

    /// Whether this node has children.
    pub fn is_expanded(&self) -> bool {
        self.expanded.load(Ordering::Relaxed)
    }

    /// Get the best child by visit count (for move selection).
    pub fn best_child_by_visits(&self) -> Option<&Node> {
        self.children.as_ref()?.iter().max_by_key(|c| c.visit_count())
    }

    /// Get visit count distribution over children (for policy target).
    pub fn visit_distribution(&self, temperature: f32) -> Vec<(Move, f32)> {
        let children = match &self.children {
            Some(c) => c,
            None => return Vec::new(),
        };

        if temperature < 1e-6 {
            // Greedy: all weight on most-visited child
            let best = children.iter().max_by_key(|c| c.visit_count());
            return children
                .iter()
                .map(|c| {
                    let weight = if std::ptr::eq(c, best.unwrap()) { 1.0 } else { 0.0 };
                    (c.mv, weight)
                })
                .collect();
        }

        let visits: Vec<f32> = children
            .iter()
            .map(|c| (c.visit_count() as f32).powf(1.0 / temperature))
            .collect();
        let total: f32 = visits.iter().sum();

        if total < 1e-10 {
            let uniform = 1.0 / children.len() as f32;
            return children.iter().map(|c| (c.mv, uniform)).collect();
        }

        children
            .iter()
            .zip(visits.iter())
            .map(|(c, &v)| (c.mv, v / total))
            .collect()
    }
}

// Safety: Node uses atomic fields for all mutable state accessed across threads.
// The `children` field is only written once during expansion (behind a CAS on `expanded`),
// and then only read afterward.
unsafe impl Send for Node {}
unsafe impl Sync for Node {}
