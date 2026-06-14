from math import inf

from agents.minimax_agent import MinimaxAgent


def evaluate_minimax_move(board_state, move, color, depth, maximizing):
    """Evaluate one root Minimax move in an independent worker process."""
    agent = MinimaxAgent(color=color, depth=depth)
    next_board = board_state.clone()
    next_board.apply_move(move)
    score = agent.minimax(next_board, depth - 1, -inf, inf, not maximizing)
    return move, score, agent.states_analyzed
