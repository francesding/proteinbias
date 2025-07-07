"""Tests for the BaseModel abstract class and model interface."""

import pytest
from proteinbias.models import BaseModel


class TestModel(BaseModel):
    """Test implementation of BaseModel."""

    __test__ = False

    def __init__(self, name="test_model"):
        super().__init__(name)
        self.score_calls = []

    def score_sequence(self, sequence: str) -> float:
        """Mock scoring that returns sequence length."""
        self.score_calls.append(sequence)
        return float(len(sequence))


class TestBaseModel:
    """Test cases for the BaseModel abstract class."""

    def test_abstract_class(self):
        """Test that BaseModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseModel("test")

    def test_concrete_implementation(self):
        """Test that concrete implementations work correctly."""
        model = TestModel("test_model")
        assert model.name == "test_model"
        assert isinstance(model, BaseModel)

    def test_score_sequence_abstract(self):
        """Test that score_sequence is abstract and must be implemented."""

        class IncompleteModel(BaseModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel("incomplete")

    def test_score_sequence_implementation(self):
        """Test that score_sequence works in concrete implementation."""
        model = TestModel()
        sequence = "MKLLVLSLSLVCASVA"
        score = model.score_sequence(sequence)
        assert score == 16.0  # Length of sequence
        assert model.score_calls == [sequence]

    def test_score_sequences_batch(self):
        """Test batch scoring functionality."""
        model = TestModel()
        sequences = ["MKL", "MKLLVL", "MKLLVLSLSLVCASVA"]
        scores = model.score_sequences(sequences)

        assert len(scores) == 3
        assert scores[0] == 3.0
        assert scores[1] == 6.0
        assert scores[2] == 16.0
        assert model.score_calls == sequences

    def test_score_sequences_empty(self):
        """Test batch scoring with empty list."""
        model = TestModel()
        scores = model.score_sequences([])
        assert scores == []
        assert model.score_calls == []

    def test_get_model_info(self):
        """Test that get_model_info returns correct metadata."""
        model = TestModel("my_test_model")
        info = model.get_model_info()

        assert info["name"] == "my_test_model"
        assert info["type"] == "TestModel"
        assert "test_models" in info["module"]

    def test_model_name_initialization(self):
        """Test that model name is properly initialized."""
        model = TestModel("custom_name")
        assert model.name == "custom_name"

    def test_score_sequences_single_item(self):
        """Test batch scoring with single sequence."""
        model = TestModel()
        sequences = ["PROTEIN"]
        scores = model.score_sequences(sequences)

        assert len(scores) == 1
        assert scores[0] == 7.0
        assert model.score_calls == ["PROTEIN"]


class MockBatchModel(BaseModel):
    """Mock model with optimized batch processing."""

    def __init__(self, name="batch_model"):
        super().__init__(name)
        self.batch_calls = []
        self.single_calls = []

    def score_sequence(self, sequence: str) -> float:
        """Mock single sequence scoring."""
        self.single_calls.append(sequence)
        return float(len(sequence))

    def score_sequences(self, sequences: list[str]) -> list[float]:
        """Mock batch scoring that's more efficient."""
        self.batch_calls.append(sequences)
        return [float(len(seq)) for seq in sequences]


class TestBatchProcessing:
    """Test batch processing optimization."""

    def test_batch_override(self):
        """Test that batch processing can be overridden."""
        model = MockBatchModel()
        sequences = ["MKL", "PROTEIN", "SEQUENCE"]
        scores = model.score_sequences(sequences)

        assert len(scores) == 3
        assert scores == [3.0, 7.0, 8.0]
        assert model.batch_calls == [sequences]
        assert model.single_calls == []  # Should not call single method
