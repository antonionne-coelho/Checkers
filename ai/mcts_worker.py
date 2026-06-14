import math
import random

from agents.mcts_agent import MCTSAgent
from ai.mcts_node import MCTSNode


def run_mcts_simulations(
    board_state, simulations_count, exploration_constant, seed=None
):
    """Run independent MCTS simulations in a worker process.

    Args:
        board_state: A picklable `Board` instance representing the current root state.
        simulations_count: Number of MCTS iterations to execute.
        exploration_constant: Exploration weight used in UCT selection.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of tuples `(Move, visit_count)` representing the root children visit counts.
    """
    if seed is not None:
        random.seed(seed)

    agent = MCTSAgent(
        color=board_state.current_player,
        iterations=simulations_count,
        exploration_constant=exploration_constant,
    )
    root = MCTSNode(board_state.clone())

    for _ in range(simulations_count):
        node = agent.selection(root)
        if not node.board.is_terminal():
            node = agent.expansion(node)
        result = agent.simulation(node.board)
        agent.backpropagation(node, result)

    return [(child.move, child.visits) for child in root.children]


def run_mcts_tree_worker(
    root,
    exploration_constant,
    simulations_count,
    seed=None,
    color="BLACK",
):
    """Run MCTS iterations with a shared tree root between processes."""
    if seed is not None:
        random.seed(seed)

    for _ in range(simulations_count):
        node = _selection(root, exploration_constant)
        if not _is_terminal(node):
            node = _expansion(node)
        result = _simulation(node, color)
        _backpropagation(node, result)


def _create_shared_node(node, board, move, parent):
    manager = node._manager
    child = manager.Namespace()
    child.board = board
    child.move = move
    child.parent = parent
    child.children = manager.list([])
    child.visits = manager.Value("i", 0)
    child.wins = manager.Value("d", 0.0)
    child.untried_moves = manager.list(board.get_valid_moves(board.current_player))
    child.lock = manager.Lock()
    return child


def _selection(node, exploration_constant):
    while True:
        with node.lock:
            if len(node.untried_moves) > 0:
                return node
            children = list(node.children)
            parent_visits = node.visits.value

        if not children:
            return node

        node = max(
            children,
            key=lambda child: _uct_score(child, parent_visits, exploration_constant),
        )


def _uct_score(child, parent_visits, exploration_constant):
    with child.lock:
        visits = child.visits.value
        wins = child.wins.value
    q = wins / visits if visits > 0 else 0.0
    u = exploration_constant * math.sqrt(max(1, parent_visits)) / (1 + visits)
    return q + u


def _expansion(node):
    with node.lock:
        if len(node.untried_moves) == 0:
            return node
        index = random.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(index)

    board_copy = node.board.clone()
    board_copy.apply_move(move)
    child = _create_shared_node(node, board_copy, move, node)
    with node.lock:
        node.children.append(child)

    return child


def _is_terminal(node):
    return node.board.is_terminal()


def _select_rollout_move(board, moves):
    capture_moves = [move for move in moves if move.captured_positions]
    if capture_moves:
        best_captures = max(len(move.captured_positions) for move in capture_moves)
        best = [
            move
            for move in capture_moves
            if len(move.captured_positions) == best_captures
        ]
        return random.choice(best)
    return random.choice(moves)


def _simulation(node, color):
    board = node.board.clone()
    while not board.is_terminal():
        moves = board.get_valid_moves(board.current_player)
        if not moves:
            break
        move = _select_rollout_move(board, moves)
        board.apply_move(move)

    winner = board.get_winner()
    return 1.0 if winner == color else 0.0


def _backpropagation(node, result):
    while node is not None:
        with node.lock:
            node.visits.value += 1
            node.wins.value += result
            node = node.parent
