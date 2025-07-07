"""Tests for DatasetConfig and BenchmarkRunner functionality."""

import pandas as pd
import tempfile
import os
from unittest.mock import patch
from proteinbias.benchmark import DatasetConfig, BenchmarkRunner
from proteinbias.models import BaseModel
from proteinbias.cache import ModelScoreCache


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, name="mock_model", return_value=1.0):
        super().__init__(name)
        self.return_value = return_value
        self.called_sequences = []

    def score_sequence(self, sequence: str) -> float:
        self.called_sequences.append(sequence)
        return self.return_value


class TestDatasetConfig:
    """Test cases for DatasetConfig."""

    def test_get_dataset(self):
        """Test getting the default dataset."""
        dataset_config = DatasetConfig.get_dataset()
        assert dataset_config["name"] == "Common SwissProt Proteins Dataset"
        assert "baseline_models" in dataset_config
        assert isinstance(dataset_config["baseline_models"], list)
        assert len(dataset_config["baseline_models"]) > 0


class TestBenchmarkRunnerBasics:
    """Test basic BenchmarkRunner functionality."""

    def test_runner_initialization(self):
        """Test BenchmarkRunner initialization."""
        runner = BenchmarkRunner()
        assert runner.cache is not None

    def test_get_model_scores_no_cache(self):
        """Test getting model scores without cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete test environment
            sequences_file = os.path.join(tmpdir, "sequences.csv")
            scores_file = os.path.join(tmpdir, "scores.csv")

            # Create sequences dataset
            sequences_data = pd.DataFrame(
                {
                    "sequence": ["MKLLVL", "PROTEIN", "SEQUENCE"],
                    "genus_species": ["species_A", "species_B", "species_A"],
                    "first_protein_name": ["protein1", "protein1", "protein2"],
                    "Entry": ["P001", "P002", "P003"],
                }
            )
            sequences_data.to_csv(sequences_file, index=False)

            # Create baseline scores
            scores_data = pd.DataFrame(
                {
                    "Entry": ["P001", "P002", "P003"],
                    "baseline_model": [1.0, 2.0, 1.5],
                }
            )
            scores_data.to_csv(scores_file, index=False)

            # Setup runner
            runner = BenchmarkRunner()

            # Mock the dataset loading to use our test files
            def mock_load_dataset():
                return pd.read_csv(sequences_file).merge(
                    pd.read_csv(scores_file), on="Entry", how="inner"
                )

            runner._load_dataset = mock_load_dataset

            # Load test dataset
            df = runner._load_dataset()

            # Create mock model
            mock_model = MockModel("test_model", return_value=5.0)

            scores = runner._get_model_scores(
                mock_model, df, use_cache=False, cache_scores=False
            )

            assert len(scores) == 3
            assert all(score == 5.0 for score in scores)
            assert len(mock_model.called_sequences) == 3

    def test_get_model_scores_with_cache(self):
        """Test getting model scores with cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete test environment
            sequences_file = os.path.join(tmpdir, "sequences.csv")
            scores_file = os.path.join(tmpdir, "scores.csv")
            cache_file = os.path.join(tmpdir, "cache.csv")

            # Create sequences dataset
            sequences_data = pd.DataFrame(
                {
                    "sequence": ["MKLLVL", "PROTEIN", "SEQUENCE"],
                    "genus_species": ["species_A", "species_B", "species_A"],
                    "first_protein_name": ["protein1", "protein1", "protein2"],
                    "Entry": ["P001", "P002", "P003"],
                }
            )
            sequences_data.to_csv(sequences_file, index=False)

            # Create baseline scores
            scores_data = pd.DataFrame(
                {
                    "Entry": ["P001", "P002", "P003"],
                    "baseline_model": [1.0, 2.0, 1.5],
                }
            )
            scores_data.to_csv(scores_file, index=False)

            # Setup runner and cache
            cache = ModelScoreCache(cache_file, load_community=False)
            runner = BenchmarkRunner(cache)

            # Pre-populate cache
            cache.add_scores(
                sequence_ids=["P001", "P002"],
                scores=[3.0, 4.0],
                model_name="test_model",
            )

            # Mock the dataset loading to use our test files
            def mock_load_dataset():
                return pd.read_csv(sequences_file).merge(
                    pd.read_csv(scores_file), on="Entry", how="inner"
                )

            runner._load_dataset = mock_load_dataset

            # Load test dataset
            df = runner._load_dataset()

            # Create mock model
            mock_model = MockModel("test_model", return_value=5.0)

            scores = runner._get_model_scores(
                mock_model, df, use_cache=True, cache_scores=True
            )

            assert len(scores) == 3
            assert scores[0] == 3.0  # From cache
            assert scores[1] == 4.0  # From cache
            assert scores[2] == 5.0  # Computed
            assert len(mock_model.called_sequences) == 1  # Only one sequence computed

    @patch("proteinbias.benchmark.get_replicate_ratings_parallel")
    @patch("proteinbias.benchmark.consolidate_replicate_ratings")
    def test_run_baseline_benchmark(self, mock_consolidate, mock_get_ratings):
        """Test running baseline benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete test environment
            sequences_file = os.path.join(tmpdir, "sequences.csv")
            scores_file = os.path.join(tmpdir, "scores.csv")

            # Create sequences dataset
            sequences_data = pd.DataFrame(
                {
                    "sequence": ["MKLLVL", "PROTEIN", "SEQUENCE"],
                    "genus_species": ["species_A", "species_B", "species_A"],
                    "first_protein_name": ["protein1", "protein1", "protein2"],
                    "Entry": ["P001", "P002", "P003"],
                }
            )
            sequences_data.to_csv(sequences_file, index=False)

            # Create baseline scores
            scores_data = pd.DataFrame(
                {
                    "Entry": ["P001", "P002", "P003"],
                    "baseline_model1": [1.0, 2.0, 1.5],
                    "baseline_model2": [1.2, 1.8, 1.3],
                }
            )
            scores_data.to_csv(scores_file, index=False)

            # Setup runner
            runner = BenchmarkRunner()

            # Mock the dataset loading and config to use our test files
            def mock_load_dataset():
                return pd.read_csv(sequences_file).merge(
                    pd.read_csv(scores_file), on="Entry", how="inner"
                )

            runner._load_dataset = mock_load_dataset

            # Mock DatasetConfig to return our test models
            original_get_dataset = DatasetConfig.get_dataset
            DatasetConfig.get_dataset = lambda: {
                "baseline_models": ["baseline_model1", "baseline_model2"]
            }

            try:
                # Mock return values
                mock_get_ratings.return_value = [
                    {"baseline_model1": [1500, 1600], "baseline_model2": [1400, 1500]}
                ]
                mock_consolidate.return_value = pd.DataFrame(
                    {
                        "model": ["baseline_model1", "baseline_model2"],
                        "score": [1550, 1450],
                    }
                )

                results = runner.run_baseline_benchmark(num_replicates=2)

                # Verify mocks were called
                mock_get_ratings.assert_called_once()
                mock_consolidate.assert_called_once()

                assert isinstance(results, pd.DataFrame)
            finally:
                # Restore original method
                DatasetConfig.get_dataset = original_get_dataset


class TestCacheIntegration:
    """Test cache integration with BenchmarkRunner."""

    def test_add_scores_from_file(self):
        """Test adding scores from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup complete test environment
            sequences_file = os.path.join(tmpdir, "sequences.csv")
            baseline_scores_file = os.path.join(tmpdir, "baseline_scores.csv")
            cache_file = os.path.join(tmpdir, "cache.csv")
            scores_file = os.path.join(tmpdir, "scores.csv")

            # Create sequences dataset
            sequences_data = pd.DataFrame(
                {
                    "sequence": ["MKLLVL", "PROTEIN", "SEQUENCE"],
                    "genus_species": ["species_A", "species_B", "species_A"],
                    "first_protein_name": ["protein1", "protein1", "protein2"],
                    "Entry": ["P001", "P002", "P003"],
                }
            )
            sequences_data.to_csv(sequences_file, index=False)

            # Create baseline scores
            baseline_scores_data = pd.DataFrame(
                {
                    "Entry": ["P001", "P002", "P003"],
                    "baseline_model": [1.0, 2.0, 1.5],
                }
            )
            baseline_scores_data.to_csv(baseline_scores_file, index=False)

            # Create scores file
            scores_df = pd.DataFrame(
                {"sequence_id": ["P001", "P002", "P003"], "score": [1.1, 2.2, 3.3]}
            )
            scores_df.to_csv(scores_file, index=False)

            # Setup runner and cache
            cache = ModelScoreCache(cache_file, load_community=False)
            runner = BenchmarkRunner(cache)

            # Mock DatasetConfig to use our test files
            original_get_dataset = DatasetConfig.get_dataset
            DatasetConfig.get_dataset = lambda: {
                "sequences_path": sequences_file,
                "baseline_scores_path": baseline_scores_file,
                "baseline_models": ["baseline_model"],
            }

            # Mock the dataset loading to use our test files
            def mock_load_dataset():
                return pd.read_csv(sequences_file).merge(
                    pd.read_csv(baseline_scores_file), on="Entry", how="inner"
                )

            runner._load_dataset = mock_load_dataset

            try:
                success = runner.add_scores_from_file(scores_file, "test_model")

                assert success
                cached_scores = cache.get_scores(["P001", "P002", "P003"], "test_model")
                assert len(cached_scores) == 3
                assert cached_scores["P001"] == 1.1
            finally:
                # Restore original method
                DatasetConfig.get_dataset = original_get_dataset
