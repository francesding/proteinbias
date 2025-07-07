"""Example model implementation for protein bias benchmarking."""

import math
from proteinbias import BaseModel


class SequenceLengthModel(BaseModel):
    """Model that scores based on sequence length."""

    def __init__(self):
        super().__init__("SequenceLengthModel")

    def score_sequence(self, sequence: str) -> float:
        """Score based on sequence length."""
        return math.log(len(sequence))
