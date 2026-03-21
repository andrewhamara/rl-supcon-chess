"""GPU-accelerated self-play with batched MCTS across parallel games.

Runs N games simultaneously. Each MCTS simulation round collects one leaf
per game, batches them into a single GPU forward pass, then expands and
backpropagates. This maximizes GPU utilization.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    import chess_engine_py as engine
    # Check that the board-ops bindings are available (added after initial build)
    _HAS_ENGINE = hasattr(engine, "is_game_over")
except ImportError:
    _HAS_ENGINE = False

if not _HAS_ENGINE:
    import chess  # python-chess fallback


# ── Board abstraction ─────────────────────────────────────────────────
# Uses Rust bindings when available, python-chess otherwise.

if _HAS_ENGINE:
    def _encode_board(fen: str) -> list[float]:
        return engine.encode_board(fen)

    def _legal_moves_indexed(fen: str) -> list[tuple[str, int]]:
        return engine.legal_moves_indexed(fen)

    def _make_move(fen: str, uci: str) -> str:
        return engine.make_move(fen, uci)

    def _is_game_over(fen: str) -> bool:
        return engine.is_game_over(fen)

    def _game_result(fen: str) -> float:
        return engine.game_result(fen)
else:
    # Pure-Python fallback using python-chess
    # Move encoding tables (must match Rust policy.rs exactly)
    # The Rust encoding: for each source square (0-63, a1=0 row-major):
    #   56 queen-move slots: 8 directions × 7 distances (only valid ones get real moves)
    #   8 knight-move slots: fixed order
    #   9 underpromotion slots: 3 pieces × 3 directions
    # Total slots per square = 73, but only reachable moves are assigned.
    # The index is: source_square * 73 + move_type_offset, filtered to valid only → 1858.

    _MOVE_TO_INDEX: dict[tuple[str, str], int] | None = None

    # Must match Rust policy.rs build_tables() exactly:
    # 1. Queen moves: for each square (rank-major), 8 directions, distances 1-7
    #    BREAK when hitting board edge (not skip!) — idx only increments for valid
    # 2. Knight moves: for each square, 8 deltas, SKIP invalid (continue)
    # 3. Underpromotions: for each file, 3 file-deltas × 3 pieces, SKIP invalid
    _QUEEN_DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    _KNIGHT_DELTAS = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    _PROMO_FILE_DELTAS = [-1, 0, 1]
    _PROMO_PIECES = ['n', 'b', 'r']

    def _sq_name(file: int, rank: int) -> str:
        return chr(ord('a') + file) + chr(ord('1') + rank)

    def _build_move_table():
        global _MOVE_TO_INDEX
        if _MOVE_TO_INDEX is not None:
            return
        _MOVE_TO_INDEX = {}
        idx = 0

        # Phase 1: Queen moves (matches Rust ordering exactly)
        for from_rank in range(8):
            for from_file in range(8):
                for dr, df in _QUEEN_DIRS:
                    for dist in range(1, 8):
                        to_rank = from_rank + dr * dist
                        to_file = from_file + df * dist
                        if to_rank < 0 or to_rank > 7 or to_file < 0 or to_file > 7:
                            break  # BREAK, not continue!
                        src = _sq_name(from_file, from_rank)
                        dst = _sq_name(to_file, to_rank)
                        uci = src + dst
                        _MOVE_TO_INDEX[("w", uci)] = idx
                        # Black perspective: flip
                        f_src = _sq_name(7 - from_file, 7 - from_rank)
                        f_dst = _sq_name(7 - to_file, 7 - to_rank)
                        _MOVE_TO_INDEX[("b", f_src + f_dst)] = idx
                        # Queen promotions encoded here too
                        if from_rank == 6 and to_rank == 7:
                            _MOVE_TO_INDEX[("w", uci + "q")] = idx
                            _MOVE_TO_INDEX[("b", f_src + f_dst + "q")] = idx
                        idx += 1

        # Phase 2: Knight moves
        for from_rank in range(8):
            for from_file in range(8):
                for kr, kf in _KNIGHT_DELTAS:
                    to_rank = from_rank + kr
                    to_file = from_file + kf
                    if to_rank < 0 or to_rank > 7 or to_file < 0 or to_file > 7:
                        continue  # skip invalid, don't increment idx
                    src = _sq_name(from_file, from_rank)
                    dst = _sq_name(to_file, to_rank)
                    uci = src + dst
                    _MOVE_TO_INDEX[("w", uci)] = idx
                    f_src = _sq_name(7 - from_file, 7 - from_rank)
                    f_dst = _sq_name(7 - to_file, 7 - to_rank)
                    _MOVE_TO_INDEX[("b", f_src + f_dst)] = idx
                    idx += 1

        # Phase 3: Underpromotions (from file, not from square)
        for from_file in range(8):
            for fd in _PROMO_FILE_DELTAS:
                to_file = from_file + fd
                if to_file < 0 or to_file > 7:
                    continue
                for piece_ch in _PROMO_PIECES:
                    src = _sq_name(from_file, 6)  # rank 7 (0-indexed 6) for white
                    dst = _sq_name(to_file, 7)
                    _MOVE_TO_INDEX[("w", src + dst + piece_ch)] = idx
                    # Black: from rank 1 to rank 0, flipped to perspective
                    f_src = _sq_name(7 - from_file, 6)
                    f_dst = _sq_name(7 - to_file, 7)
                    _MOVE_TO_INDEX[("b", f_src + f_dst + piece_ch)] = idx
                    idx += 1

        assert idx == 1858, f"Expected 1858 moves, got {idx}"

    def _encode_move_py(uci: str, is_white: bool) -> int:
        """Encode a UCI move to policy index."""
        _build_move_table()
        perspective = "w" if is_white else "b"
        # Queen promotions use the queen-move encoding (no promotion suffix)
        if len(uci) == 5 and uci[4] == 'q':
            key = (perspective, uci[:4])
        else:
            key = (perspective, uci)
        return _MOVE_TO_INDEX.get(key, 0)

    def _encode_board(fen: str) -> list[float]:
        """Encode board features matching Rust board.rs."""
        board = chess.Board(fen)
        features = [0.0] * (21 * 64)
        perspective = board.turn  # True = white

        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if not perspective:  # black perspective: flip board
                rank = 7 - rank
                file = 7 - file

            is_friendly = (piece.color == perspective)
            color_offset = 0 if is_friendly else 6
            piece_type_idx = piece.piece_type - 1  # 1-6 -> 0-5
            plane = color_offset + piece_type_idx
            features[plane * 64 + rank * 8 + file] = 1.0

        # Plane 12: side to move
        if perspective:
            for i in range(64):
                features[12 * 64 + i] = 1.0

        # Planes 13-16: castling
        if board.has_kingside_castling_rights(chess.WHITE):
            for i in range(64): features[13 * 64 + i] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            for i in range(64): features[14 * 64 + i] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            for i in range(64): features[15 * 64 + i] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            for i in range(64): features[16 * 64 + i] = 1.0

        # Plane 17: en passant
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            if not perspective:
                ep_rank = 7 - ep_rank
                ep_file = 7 - ep_file
            features[17 * 64 + ep_rank * 8 + ep_file] = 1.0

        # Plane 18: halfmove clock
        hmc = min(board.halfmove_clock, 100) / 100.0
        for i in range(64): features[18 * 64 + i] = hmc

        # Plane 19: fullmove number
        fmn = min(board.fullmove_number, 200) / 200.0
        for i in range(64): features[19 * 64 + i] = fmn

        # Plane 20: all ones
        for i in range(64): features[20 * 64 + i] = 1.0

        return features

    def _legal_moves_indexed(fen: str) -> list[tuple[str, int]]:
        board = chess.Board(fen)
        is_white = board.turn
        result = []
        for mv in board.legal_moves:
            uci = mv.uci()
            idx = _encode_move_py(uci, is_white)
            result.append((uci, idx))
        return result

    def _make_move(fen: str, uci: str) -> str:
        board = chess.Board(fen)
        board.push_uci(uci)
        return board.fen()

    def _is_game_over(fen: str) -> bool:
        board = chess.Board(fen)
        return board.is_game_over(claim_draw=True)

    def _game_result(fen: str) -> float:
        board = chess.Board(fen)
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if board.turn == chess.WHITE else 1.0
        return 0.0


# ── MCTS Node ────────────────────────────────────────────────────────

class MctsNode:
    """Lightweight MCTS node."""
    __slots__ = ["visit_count", "total_value", "prior", "children",
                 "move_uci", "policy_idx"]

    def __init__(self, prior: float = 0.0, move_uci: str = "", policy_idx: int = 0):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children: list["MctsNode"] = []
        self.move_uci = move_uci
        self.policy_idx = policy_idx

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def best_child(self, c_puct: float) -> "MctsNode":
        """Select child with highest PUCT score."""
        parent_sqrt = math.sqrt(self.visit_count)
        best = None
        best_score = -1e9
        for child in self.children:
            exploit = child.q_value
            explore = c_puct * child.prior * parent_sqrt / (1 + child.visit_count)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self, legal_moves: list[tuple[str, int]], policy_probs: np.ndarray,
               dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.0):
        """Expand node with children for all legal moves."""
        priors = []
        for uci, idx in legal_moves:
            priors.append(policy_probs[idx])

        # Normalize priors over legal moves
        prior_sum = sum(priors)
        if prior_sum > 1e-8:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(legal_moves)] * len(legal_moves)

        # Dirichlet noise at root
        if dirichlet_alpha > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            priors = [
                (1 - dirichlet_epsilon) * p + dirichlet_epsilon * n
                for p, n in zip(priors, noise)
            ]

        for (uci, idx), prior in zip(legal_moves, priors):
            self.children.append(MctsNode(prior=prior, move_uci=uci, policy_idx=idx))

    def visit_distribution(self, temperature: float) -> list[tuple[int, float]]:
        """Get visit count distribution over children for policy target."""
        if temperature < 0.01:
            # Greedy
            best = max(self.children, key=lambda c: c.visit_count)
            return [(c.policy_idx, 1.0 if c is best else 0.0) for c in self.children]

        visits = np.array([c.visit_count for c in self.children], dtype=np.float64)
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total < 1e-8:
            visits = np.ones(len(self.children)) / len(self.children)
        else:
            visits /= total

        return [(c.policy_idx, float(v)) for c, v in zip(self.children, visits)]


# ── Game State ────────────────────────────────────────────────────────

class GameState:
    """Tracks a single in-progress self-play game."""

    def __init__(self):
        self.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.root: MctsNode | None = None
        self.positions: list[tuple[np.ndarray, list[tuple[int, float]], str]] = []
        # (features, policy_target, fen_at_position) -- fen needed for side-to-move
        self.move_number = 0
        self.done = False
        self.outcome = 0.0  # from white's perspective

        # MCTS state for current search
        self.sim_path: list[MctsNode] = []
        self.leaf_fen: str = ""
        self.needs_eval = False

    @property
    def side_to_move_is_white(self) -> bool:
        return " w " in self.fen


# ── Batched Self-Play ─────────────────────────────────────────────────

def gpu_selfplay(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    num_simulations: int = 200,
    c_puct: float = 2.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    max_moves: int = 512,
    callback=None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run self-play games with batched GPU inference.

    Args:
        model: PyTorch ChessNet on device.
        device: torch device (cuda/mps).
        num_games: Number of games to play simultaneously.
        callback: Optional callable(games_completed, total_games) for progress.

    Returns:
        List of (features[21,8,8], policy[1858], value) training positions.
    """
    model.eval()
    games = [GameState() for _ in range(num_games)]
    all_positions = []
    completed = 0

    while True:
        # Collect active games that need a move
        active = [g for g in games if not g.done]
        if not active:
            break

        # ── For each active game, run MCTS ──
        for g in active:
            if _is_game_over(g.fen) or g.move_number >= max_moves:
                _finish_game(g)
                completed += 1
                if callback:
                    callback(completed, num_games)
                continue

            # Run full MCTS search for this game's current position
            root = _run_mcts(
                model, device, g.fen,
                num_simulations, c_puct,
                dirichlet_alpha, dirichlet_epsilon,
            )

            # Record position
            features_flat = _encode_board(g.fen)
            features = np.array(features_flat, dtype=np.float32).reshape(21, 8, 8)

            temp = temperature if g.move_number < temp_threshold else 0.01
            visit_dist = root.visit_distribution(temp)
            policy_target = [(idx, prob) for idx, prob in visit_dist if prob > 1e-6]

            g.positions.append((features, policy_target, g.fen))

            # Select move proportional to visits
            if temp < 0.02:
                best_child = max(root.children, key=lambda c: c.visit_count)
            else:
                visits = np.array([c.visit_count for c in root.children], dtype=np.float64)
                visits = visits ** (1.0 / temp)
                visits /= visits.sum()
                chosen_idx = np.random.choice(len(root.children), p=visits)
                best_child = root.children[chosen_idx]

            g.fen = _make_move(g.fen, best_child.move_uci)
            g.move_number += 1

        # Check newly finished games
        for g in active:
            if not g.done and (_is_game_over(g.fen) or g.move_number >= max_moves):
                _finish_game(g)
                completed += 1
                if callback:
                    callback(completed, num_games)

    # Collect all training positions
    for g in games:
        for features, sparse_policy, fen in g.positions:
            # Build dense policy
            policy = np.zeros(1858, dtype=np.float32)
            for idx, prob in sparse_policy:
                policy[idx] = prob

            # Value from side-to-move perspective
            is_white = " w " in fen
            value = g.outcome if is_white else -g.outcome

            all_positions.append((features, policy, value))

    return all_positions


def _finish_game(g: GameState):
    """Mark game as done and compute outcome."""
    g.done = True
    if _is_game_over(g.fen):
        result = _game_result(g.fen)
        # game_result is from side-to-move's perspective; convert to white's
        if g.side_to_move_is_white:
            g.outcome = result
        else:
            g.outcome = -result
    else:
        g.outcome = 0.0  # draw by move limit


def _run_mcts(
    model: torch.nn.Module,
    device: torch.device,
    root_fen: str,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> MctsNode:
    """Run MCTS from a position, using GPU for leaf evaluation.

    Batches multiple leaf evaluations per search by collecting pending
    leaves and running them in a single forward pass.
    """
    root = MctsNode()
    BATCH_SIZE = 16  # leaves to collect before GPU eval

    # Expand root
    legal = _legal_moves_indexed(root_fen)
    if not legal:
        return root
    root_policy, root_value = _evaluate_single(model, device, root_fen)
    root.expand(legal, root_policy, dirichlet_alpha, dirichlet_epsilon)
    root.visit_count = 1
    root.total_value = root_value

    sim = 0
    while sim < num_simulations:
        # Collect a batch of leaves
        batch_paths: list[list[MctsNode]] = []
        batch_fens: list[str] = []

        collect_count = min(BATCH_SIZE, num_simulations - sim)
        for _ in range(collect_count):
            path, leaf_fen = _select_leaf(root, root_fen, c_puct)
            if leaf_fen is not None:
                batch_paths.append(path)
                batch_fens.append(leaf_fen)
            else:
                # Terminal node, backprop immediately
                _backpropagate(path, 0.0)
                sim += 1

        if not batch_fens:
            sim += collect_count
            continue

        # Batch GPU evaluation
        policies, values = _evaluate_batch(model, device, batch_fens)

        for path, fen, policy, value in zip(batch_paths, batch_fens, policies, values):
            leaf = path[-1]
            # Expand leaf
            leaf_legal = _legal_moves_indexed(fen)
            if leaf_legal and leaf.visit_count == 0:
                leaf.expand(leaf_legal, policy)

            # Backprop (negate value since it's from leaf's perspective)
            _backpropagate(path, value)
            sim += 1

    return root


def _select_leaf(
    root: MctsNode, root_fen: str, c_puct: float
) -> tuple[list[MctsNode], str | None]:
    """Walk from root to a leaf, applying virtual loss. Returns (path, leaf_fen)."""
    node = root
    fen = root_fen
    path = [node]

    while node.children:
        node = node.best_child(c_puct)
        path.append(node)
        # Apply virtual loss
        node.visit_count += 1
        node.total_value -= 1.0
        fen = _make_move(fen, node.move_uci)

    # Check if terminal
    if _is_game_over(fen):
        result = _game_result(fen)
        _backpropagate(path, result)
        # Undo virtual loss effect (backprop already added visits)
        return path, None

    return path, fen


def _backpropagate(path: list[MctsNode], value: float):
    """Backpropagate value up the path, alternating perspective."""
    for i, node in enumerate(reversed(path)):
        # Undo virtual loss (added +1 visit, -1 value during select)
        if i > 0:  # skip root, which didn't get virtual loss
            node.visit_count -= 1
            node.total_value += 1.0
        # Now apply real update
        node.visit_count += 1
        sign = 1.0 if i % 2 == 0 else -1.0
        node.total_value += sign * value


@torch.no_grad()
def _evaluate_single(
    model: torch.nn.Module, device: torch.device, fen: str
) -> tuple[np.ndarray, float]:
    """Evaluate a single position on GPU."""
    features = np.array(_encode_board(fen), dtype=np.float32).reshape(1, 21, 8, 8)
    x = torch.from_numpy(features).to(device)
    policy_logits, value_pred, _, _ = model(x)
    policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()
    value = value_pred[0].item()
    return policy, value


@torch.no_grad()
def _evaluate_batch(
    model: torch.nn.Module, device: torch.device, fens: list[str]
) -> tuple[list[np.ndarray], list[float]]:
    """Evaluate a batch of positions on GPU in a single forward pass."""
    batch = np.zeros((len(fens), 21, 8, 8), dtype=np.float32)
    for i, fen in enumerate(fens):
        feats = _encode_board(fen)
        batch[i] = np.array(feats, dtype=np.float32).reshape(21, 8, 8)

    x = torch.from_numpy(batch).to(device)
    policy_logits, value_pred, _, _ = model(x)
    policies = F.softmax(policy_logits, dim=1).cpu().numpy()
    values = value_pred.squeeze(-1).cpu().tolist()

    return [policies[i] for i in range(len(fens))], values
