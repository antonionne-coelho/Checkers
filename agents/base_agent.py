from abc import ABC, abstractmethod


class Agent(ABC):
    """Base abstract class for all game agents."""

    def __init__(self, name: str, color: str):
        """Initialize the agent with a name and color."""
        self.name = name
        self.color = color
        self.last_decision_info = {}

    @abstractmethod
    def get_best_move(self, board):
        """Return the best move for the current board state."""
        pass

