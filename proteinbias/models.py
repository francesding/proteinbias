"""Base model interface for protein sequence scoring."""

import abc
from typing import List, Dict, Any


class BaseModel(abc.ABC):
    """Abstract base class for protein sequence scoring models.

    Users should subclass this to implement their own models for benchmarking.
    """

    def __init__(self, name: str):
        """Initialize the model.

        Args:
            name: Human-readable name for the model (used in results)
        """
        self.name = name

    @abc.abstractmethod
    def score_sequence(self, sequence: str) -> float:
        """Score a single protein sequence.

        Args:
            sequence: Protein sequence string (amino acids)

        Returns:
            float: Model score for the sequence (higher is better)
        """
        pass

    def score_sequences(self, sequences: List[str]) -> List[float]:
        """Score multiple protein sequences.

        Default implementation calls score_sequence for each sequence.
        Override this method if your model supports batch processing.

        Args:
            sequences: List of protein sequence strings

        Returns:
            List of model scores for the sequences
        """
        return [self.score_sequence(seq) for seq in sequences]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model for result tracking.

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
