"""Example model implementation for protein bias benchmarking."""

import random
import math
from proteinbias import BaseModel


class RandomModel(BaseModel):
    """Simple random model for testing."""

    def __init__(self):
        super().__init__("RandomModel")

    def score_sequence(self, sequence: str) -> float:
        """Return a random score based on sequence length."""
        # Simple random scoring with some bias toward longer sequences
        random.seed(hash(sequence) % 2**32)
        base_score = random.uniform(-2.0, 0.0)
        length_bonus = math.log(len(sequence)) * 0.1
        return base_score + length_bonus


class SequenceLengthModel(BaseModel):
    """Model that scores based on sequence length."""

    def __init__(self):
        super().__init__("SequenceLengthModel")

    def score_sequence(self, sequence: str) -> float:
        """Score based on sequence length."""
        return math.log(len(sequence))


class AminoAcidCompositionModel(BaseModel):
    """Model that scores based on amino acid composition."""

    def __init__(self):
        super().__init__("AminoAcidCompositionModel")
        # Hydrophobic amino acids
        self.hydrophobic = set("AILMFPWV")

    def score_sequence(self, sequence: str) -> float:
        """Score based on hydrophobic amino acid content."""
        if not sequence:
            return float("-inf")

        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic)
        hydrophobic_fraction = hydrophobic_count / len(sequence)

        return hydrophobic_fraction


if __name__ == "__main__":
    # Test the example models
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRIPMGKMYLHOQGAYVWDEVVTLNVVKEKDMFQETIDLETLNEDLERSAWVKSFDNSYFKELLSAGEFLKIYSLNFMRGYFIRDIIKEFKSLPSEEIQGPPEEELASTYSKGDMAEISLLALLSILKHTQVWKKLKDQWMVDWKQAEKEWLVLVNQGPSYGPVDDRQEQETRSSYAETLKQPKLIHSEAKFHVLHIRNKPDVVDVWKDLAKDDQHIIVTDQDHSLDQRKEKVLFFHAAAKFVFMVQPAIIGGFYQRSPGGDRDKYMLELEKGDMFRNPMSKAVIEEKVKRRFQSQPQDEAIIGQVVNSFPRRFSDWMKPSLDLTFQVVMQYEGDGKFMQVKLYRFGDTDNMQSAAFGDKSNFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQGFGDKSFPKSFQRFGHKHVKEQETDGVDMQFQDALLKQVPKLMKGTKQTNDNLPQDNLRLDQFGDTDQFHQ"

    models = [
        RandomModel(),
        SequenceLengthModel(),
        AminoAcidCompositionModel(),
    ]

    for model in models:
        score = model.score_sequence(test_sequence)
        print(f"{model.name}: {score:.4f}")
