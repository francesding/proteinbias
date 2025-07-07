"""Tests for Elo computation functions."""

import pytest
import pandas as pd
from proteinbias.compute_elo import (
    expected_outcome,
    update_elo,
    EloBenchmarker,
    process_single_replicate,
    consolidate_replicate_ratings,
)


class TestEloFunctions:
    """Test basic Elo computation functions."""

    def test_expected_outcome_equal_ratings(self):
        """Test expected outcome with equal ratings."""
        result = expected_outcome(1500.0, 1500.0)
        assert abs(result - 0.5) < 1e-10

    def test_expected_outcome_higher_rating(self):
        """Test expected outcome with higher rating."""
        result = expected_outcome(1600.0, 1500.0)
        assert result > 0.5
        assert result < 1.0

    def test_expected_outcome_lower_rating(self):
        """Test expected outcome with lower rating."""
        result = expected_outcome(1400.0, 1500.0)
        assert result < 0.5
        assert result > 0.0

    def test_expected_outcome_symmetry(self):
        """Test that expected outcomes are symmetric."""
        result1 = expected_outcome(1600.0, 1400.0)
        result2 = expected_outcome(1400.0, 1600.0)
        assert abs(result1 + result2 - 1.0) < 1e-10

    def test_update_elo_win(self):
        """Test Elo update for a win."""
        initial_rating = 1500.0
        opponent_rating = 1500.0
        new_rating = update_elo(initial_rating, opponent_rating, 1.0, k=32)
        assert new_rating > initial_rating

    def test_update_elo_loss(self):
        """Test Elo update for a loss."""
        initial_rating = 1500.0
        opponent_rating = 1500.0
        new_rating = update_elo(initial_rating, opponent_rating, 0.0, k=32)
        assert new_rating < initial_rating

    def test_update_elo_draw(self):
        """Test Elo update for a draw with equal ratings."""
        initial_rating = 1500.0
        opponent_rating = 1500.0
        new_rating = update_elo(initial_rating, opponent_rating, 0.5, k=32)
        assert abs(new_rating - initial_rating) < 1e-10

    def test_update_elo_k_factor(self):
        """Test that K-factor affects rating change magnitude."""
        initial_rating = 1500.0
        opponent_rating = 1500.0

        new_rating_k16 = update_elo(initial_rating, opponent_rating, 1.0, k=16)
        new_rating_k32 = update_elo(initial_rating, opponent_rating, 1.0, k=32)

        change_k16 = new_rating_k16 - initial_rating
        change_k32 = new_rating_k32 - initial_rating

        assert abs(change_k32) > abs(change_k16)


class TestEloBenchmarker:
    """Test the EloBenchmarker class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "first_protein_name": ["protein1", "protein1", "protein2", "protein2"],
                "genus_species": ["species_A", "species_B", "species_A", "species_B"],
                "model1_ll": [1.0, 2.0, 1.5, 2.5],
                "model2_ll": [1.2, 1.8, 1.3, 2.3],
            }
        )

    def test_benchmarker_initialization(self, sample_data):
        """Test EloBenchmarker initialization."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        assert benchmarker.models == models
        assert benchmarker.initial_rating == 1500.0
        assert benchmarker.k == 16
        assert len(benchmarker.species_medians) > 0

    def test_compute_species_medians(self, sample_data):
        """Test species median computation."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        # Check that medians are computed
        key = ("protein1", "species_A", "model1_ll")
        assert key in benchmarker.species_medians
        assert benchmarker.species_medians[key] == 1.0

    def test_get_species_pairs(self, sample_data):
        """Test species pair generation."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        pairs = benchmarker._get_species_pairs("protein1")
        assert len(pairs) == 1
        assert ("species_A", "species_B") in pairs or (
            "species_B",
            "species_A",
        ) in pairs

    def test_compute_matchup_result(self, sample_data):
        """Test matchup result computation."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        # species_B has higher score (2.0 vs 1.0) for protein1, model1_ll
        result = benchmarker._compute_matchup_result(
            "protein1", "species_A", "species_B", "model1_ll"
        )
        assert result == 0.0  # species_A loses

        result = benchmarker._compute_matchup_result(
            "protein1", "species_B", "species_A", "model1_ll"
        )
        assert result == 1.0  # species_B wins

    def test_compute_elo_single_epoch(self, sample_data):
        """Test single epoch Elo computation."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        ratings = benchmarker.compute_elo_single_epoch(random_state=42)

        # Check that ratings are returned for both models
        assert "model1_ll" in ratings
        assert "model2_ll" in ratings

        # Check that both species have ratings
        for model in models:
            assert "species_A" in ratings[model]
            assert "species_B" in ratings[model]

    def test_get_ratings_dataframe(self, sample_data):
        """Test conversion to DataFrame."""
        models = ["model1_ll", "model2_ll"]
        benchmarker = EloBenchmarker(sample_data, models=models)

        # Run one epoch to generate ratings
        benchmarker.compute_elo_single_epoch(random_state=42)

        ratings_df = benchmarker.get_ratings_dataframe()

        assert "genus_species" in ratings_df.columns
        assert "model" in ratings_df.columns
        assert "elo_rating" in ratings_df.columns
        assert len(ratings_df) == 4  # 2 models * 2 species


class TestProcessingFunctions:
    """Test parallel processing functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "first_protein_name": ["protein1", "protein1", "protein2", "protein2"],
                "genus_species": ["species_A", "species_B", "species_A", "species_B"],
                "model1_ll": [1.0, 2.0, 1.5, 2.5],
                "model2_ll": [1.2, 1.8, 1.3, 2.3],
            }
        )

    def test_process_single_replicate(self, sample_data):
        """Test single replicate processing."""
        models = ["model1_ll", "model2_ll"]
        args = (sample_data, models, 16, 42)

        result = process_single_replicate(args)

        assert isinstance(result, dict)
        assert "model1_ll" in result
        assert "model2_ll" in result

        for model in models:
            assert "species_A" in result[model]
            assert "species_B" in result[model]

    def test_consolidate_replicate_ratings(self, sample_data):
        """Test consolidation of replicate ratings."""
        models = ["model1_ll", "model2_ll"]

        # Create mock replicate ratings
        replicate1 = {
            "model1_ll": {"species_A": 1500.0, "species_B": 1520.0},
            "model2_ll": {"species_A": 1480.0, "species_B": 1540.0},
        }
        replicate2 = {
            "model1_ll": {"species_A": 1510.0, "species_B": 1510.0},
            "model2_ll": {"species_A": 1490.0, "species_B": 1530.0},
        }

        ratings_replicates = [replicate1, replicate2]

        result_df = consolidate_replicate_ratings(ratings_replicates, models)

        assert "genus_species" in result_df.columns
        assert len(result_df) == 2  # 2 species

        # Check that mean columns exist
        for model in models:
            assert f"Elo_{model}_mean" in result_df.columns
            assert f"Elo_{model}_SE" in result_df.columns

    def test_consolidate_ratings_with_missing_species(self):
        """Test consolidation when some species are missing from some replicates."""
        models = ["model1_ll"]

        # Create replicates with different species
        replicate1 = {"model1_ll": {"species_A": 1500.0, "species_B": 1520.0}}
        replicate2 = {"model1_ll": {"species_A": 1510.0, "species_C": 1490.0}}

        ratings_replicates = [replicate1, replicate2]

        result_df = consolidate_replicate_ratings(ratings_replicates, models)

        # Should have all species from both replicates
        species_list = result_df["genus_species"].tolist()
        assert "species_A" in species_list
        assert "species_B" in species_list
        assert "species_C" in species_list


class TestEloMathematicalProperties:
    """Test mathematical properties of Elo system."""

    def test_elo_conservation(self):
        """Test that Elo ratings are conserved in zero-sum games."""
        initial_rating = 1500.0
        opponent_rating = 1600.0
        k = 32

        # Test win scenario
        new_player = update_elo(initial_rating, opponent_rating, 1.0, k)
        new_opponent = update_elo(opponent_rating, initial_rating, 0.0, k)

        total_change = (new_player - initial_rating) + (new_opponent - opponent_rating)
        assert abs(total_change) < 1e-10  # Should be approximately zero

    def test_elo_expected_outcome_range(self):
        """Test that expected outcomes are always between 0 and 1."""
        test_ratings = [1000, 1200, 1500, 1800, 2000]

        for rating1 in test_ratings:
            for rating2 in test_ratings:
                outcome = expected_outcome(rating1, rating2)
                assert 0.0 <= outcome <= 1.0

    def test_elo_update_bounds(self):
        """Test that Elo updates have reasonable bounds."""
        initial_rating = 1500.0
        k = 32

        # Test extreme cases
        new_rating_best = update_elo(
            initial_rating, 0.0, 1.0, k
        )  # Win against 0-rated opponent
        new_rating_worst = update_elo(
            initial_rating, 3000.0, 0.0, k
        )  # Loss against 3000-rated opponent

        # Updates should be bounded by K-factor
        assert abs(new_rating_best - initial_rating) <= k
        assert abs(new_rating_worst - initial_rating) <= k
