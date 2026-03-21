#!/usr/bin/env python3
"""Interactive terminal chess interface — play against the engine."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PIECE_SYMBOLS = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·",
}

# Simple board representation for display (no dependency on chess_engine_py for basic play)
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class SimpleBoard:
    """Minimal board for terminal display. Uses python-chess if available, otherwise FEN parsing."""

    def __init__(self):
        try:
            import chess
            self.board = chess.Board()
            self.use_python_chess = True
        except ImportError:
            self.board = None
            self.use_python_chess = False
            self.squares = self._parse_fen(INITIAL_FEN)
            self.turn = "w"

    def _parse_fen(self, fen: str) -> list[list[str]]:
        parts = fen.split()
        rows = parts[0].split("/")
        board = []
        for row in rows:
            r = []
            for ch in row:
                if ch.isdigit():
                    r.extend(["."] * int(ch))
                else:
                    r.append(ch)
            board.append(r)
        return board

    def display(self):
        print()
        print("    a   b   c   d   e   f   g   h")
        print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

        if self.use_python_chess:
            import chess
            for rank in range(7, -1, -1):
                row_str = f"{rank + 1} │"
                for file in range(8):
                    sq = chess.square(file, rank)
                    piece = self.board.piece_at(sq)
                    if piece:
                        symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())
                    else:
                        is_dark = (rank + file) % 2 == 0
                        symbol = "·" if is_dark else " "
                    row_str += f" {symbol} │"
                print(row_str)
                if rank > 0:
                    print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
        else:
            for rank_idx, row in enumerate(self.squares):
                rank_num = 8 - rank_idx
                row_str = f"{rank_num} │"
                for file_idx, piece in enumerate(row):
                    symbol = PIECE_SYMBOLS.get(piece, piece)
                    row_str += f" {symbol} │"
                print(row_str)
                if rank_idx < 7:
                    print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

        print("  └───┴───┴───┴───┴───┴───┴───┴───┘")
        print("    a   b   c   d   e   f   g   h")
        print()

    def make_move(self, move_str: str) -> bool:
        """Apply a move in UCI format (e.g., 'e2e4'). Returns True if valid."""
        if self.use_python_chess:
            import chess
            try:
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    return True
                # Try as SAN
                try:
                    move = self.board.parse_san(move_str)
                    self.board.push(move)
                    return True
                except (chess.InvalidMoveError, chess.IllegalMoveError):
                    pass
                return False
            except (ValueError, chess.InvalidMoveError):
                return False
        else:
            print("(Move validation requires python-chess: pip install chess)")
            return True

    def is_game_over(self) -> bool:
        if self.use_python_chess:
            return self.board.is_game_over()
        return False

    def result(self) -> str:
        if self.use_python_chess:
            return self.board.result()
        return "*"

    def turn_str(self) -> str:
        if self.use_python_chess:
            return "White" if self.board.turn else "Black"
        return "White" if self.turn == "w" else "Black"

    def fen(self) -> str:
        if self.use_python_chess:
            return self.board.fen()
        return INITIAL_FEN

    def get_engine_move(self, weights_path: str | None, simulations: int) -> str | None:
        """Get a move from the Rust engine."""
        try:
            import chess_engine_py as engine
            fen = self.fen()
            wp = weights_path or ""
            move_uci = engine.search_move(fen, weights_path=wp, simulations=simulations)
            return move_uci
        except ImportError:
            return None
        except Exception as e:
            print(f"Engine error: {e}")
            return None


def engine_move_random():
    """Get a random legal move using python-chess."""
    import random
    try:
        import chess
        return None  # Let the caller handle this
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Play chess against the engine")
    parser.add_argument("--weights", type=str, default=None, help="Path to engine weights")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Your color")
    args = parser.parse_args()

    board = SimpleBoard()
    human_is_white = args.color == "white"

    print("=" * 45)
    print("    Chess Engine — AlphaZero + SupCon")
    print("=" * 45)
    print()
    print(f"You are playing as {'White' if human_is_white else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4) or SAN (e.g., Nf3)")
    print("Commands: 'quit', 'fen', 'undo'")
    print()

    board.display()

    while not board.is_game_over():
        is_human_turn = (board.turn_str() == "White") == human_is_white

        if is_human_turn:
            move_str = input(f"{board.turn_str()} to move: ").strip()

            if move_str.lower() in ("quit", "q", "exit"):
                print("Goodbye!")
                return
            if move_str.lower() == "fen":
                print(board.fen())
                continue
            if move_str.lower() == "undo":
                if board.use_python_chess and len(board.board.move_stack) >= 2:
                    board.board.pop()
                    board.board.pop()
                    board.display()
                else:
                    print("Cannot undo")
                continue
            if not move_str:
                continue

            if not board.make_move(move_str):
                print(f"Invalid move: {move_str}")
                continue
        else:
            print(f"{board.turn_str()} (engine) is thinking...")
            engine_mv = board.get_engine_move(args.weights, args.simulations)
            if engine_mv:
                board.make_move(engine_mv)
                print(f"Engine plays: {engine_mv}")
            else:
                # Fallback: random move
                if board.use_python_chess:
                    import random
                    legal = list(board.board.legal_moves)
                    if legal:
                        mv = random.choice(legal)
                        board.board.push(mv)
                        print(f"Engine plays: {mv.uci()} (random — no weights loaded)")
                    else:
                        break
                else:
                    print("Engine not available. Enter a move for both sides.")
                    move_str = input(f"{board.turn_str()} to move: ").strip()
                    if move_str.lower() in ("quit", "q"):
                        return
                    board.make_move(move_str)

        board.display()

    print(f"Game over: {board.result()}")


if __name__ == "__main__":
    main()
