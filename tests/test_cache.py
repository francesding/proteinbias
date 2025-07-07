"""Tests for ModelScoreCache functionality."""

import pytest
import pandas as pd
import tempfile
import os
from proteinbias.cache import ModelScoreCache


class TestModelScoreCache:
    """Test cases for ModelScoreCache."""

    def test_cache_initialization(self):
        """Test cache initialization with default and custom paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)
            assert cache.cache_file == cache_file
            assert len(cache.cache_df) == 0

    def test_add_scores_basic(self):
        """Test basic score addition to cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            sequence_ids = ["seq1", "seq2", "seq3"]
            scores = [1.5, 2.3, 1.8]
            model_name = "test_model"

            added = cache.add_scores(sequence_ids, scores, model_name)
            assert added == 3
            assert len(cache.cache_df) == 3

    def test_add_scores_validation(self):
        """Test input validation for add_scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Test mismatched lengths
            with pytest.raises(ValueError, match="must match number of scores"):
                cache.add_scores(["seq1", "seq2"], [1.5], "model")

            # Test empty inputs
            with pytest.raises(ValueError, match="No sequence IDs provided"):
                cache.add_scores([], [], "model")

            # Test empty model name
            with pytest.raises(ValueError, match="Model name cannot be empty"):
                cache.add_scores(["seq1"], [1.5], "")

    def test_get_scores(self):
        """Test retrieving scores from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add some scores
            sequence_ids = ["seq1", "seq2", "seq3"]
            scores = [1.5, 2.3, 1.8]
            model_name = "test_model"
            cache.add_scores(sequence_ids, scores, model_name)

            # Test retrieval
            retrieved = cache.get_scores(["seq1", "seq3", "seq4"], model_name)
            assert len(retrieved) == 2
            assert retrieved["seq1"] == 1.5
            assert retrieved["seq3"] == 1.8
            assert "seq4" not in retrieved

    def test_get_scores_nonexistent_model(self):
        """Test retrieving scores for non-existent model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            retrieved = cache.get_scores(["seq1", "seq2"], "nonexistent_model")
            assert retrieved == {}

    def test_overwrite_functionality(self):
        """Test overwriting existing scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add initial scores
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "model1")

            # Try to add same scores without overwrite
            added = cache.add_scores(["seq1", "seq2"], [1.5, 2.5], "model1")
            assert added == 0  # No new entries added

            # Add with overwrite
            added = cache.add_scores(
                ["seq1", "seq2"], [1.5, 2.5], "model1", overwrite=True
            )
            assert added == 2

            # Verify updated scores
            retrieved = cache.get_scores(["seq1", "seq2"], "model1")
            assert retrieved["seq1"] == 1.5
            assert retrieved["seq2"] == 2.5

    def test_has_model_scores(self):
        """Test checking which sequences have cached scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add some scores
            cache.add_scores(["seq1", "seq3"], [1.0, 3.0], "model1")

            # Check which sequences have scores
            has_scores = cache.has_model_scores(["seq1", "seq2", "seq3"], "model1")
            assert has_scores["seq1"] is True
            assert has_scores["seq2"] is False
            assert has_scores["seq3"] is True

    def test_list_models(self):
        """Test listing models in cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Empty cache
            models = cache.list_models()
            assert models == []

            # Add scores for multiple models
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "model1")
            cache.add_scores(["seq3"], [3.0], "model2")

            models = cache.list_models()
            assert len(models) == 2

            model_names = [m["model_name"] for m in models]
            assert "model1" in model_names
            assert "model2" in model_names

    def test_clear_model(self):
        """Test clearing scores for a specific model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add scores for multiple models
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "model1")
            cache.add_scores(["seq3"], [3.0], "model2")

            # Clear model1
            removed = cache.clear_model("model1")
            assert removed == 2

            # Verify only model2 remains
            models = cache.list_models()
            assert len(models) == 1
            assert models[0]["model_name"] == "model2"

    def test_clear_all(self):
        """Test clearing entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add scores
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "model1")
            cache.add_scores(["seq3"], [3.0], "model2")

            # Clear all
            removed = cache.clear_all()
            assert removed == 3
            assert len(cache.cache_df) == 0

            # Verify empty cache
            models = cache.list_models()
            assert models == []

    def test_add_scores_from_file(self):
        """Test adding scores from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test scores file
            scores_file = os.path.join(tmpdir, "scores.csv")
            scores_df = pd.DataFrame(
                {"sequence_id": ["seq1", "seq2", "seq3"], "score": [1.5, 2.3, 1.8]}
            )
            scores_df.to_csv(scores_file, index=False)

            # Test adding from file
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            added = cache.add_scores_from_file(scores_file, "file_model")
            assert added == 3

            # Verify scores were added
            retrieved = cache.get_scores(["seq1", "seq2", "seq3"], "file_model")
            assert len(retrieved) == 3
            assert retrieved["seq1"] == 1.5

    def test_add_scores_from_file_legacy_format(self):
        """Test adding scores from CSV file with legacy 'sequence' column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test scores file with legacy format
            scores_file = os.path.join(tmpdir, "scores.csv")
            scores_df = pd.DataFrame(
                {"sequence": ["seq1", "seq2", "seq3"], "score": [1.5, 2.3, 1.8]}
            )
            scores_df.to_csv(scores_file, index=False)

            # Test adding from file
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            added = cache.add_scores_from_file(scores_file, "legacy_model")
            assert added == 3

    def test_export_scores(self):
        """Test exporting scores to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Add some scores
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "export_model")

            # Export scores
            export_file = os.path.join(tmpdir, "exported.csv")
            success = cache.export_scores("export_model", export_file)
            assert success is True

            # Verify exported file
            exported_df = pd.read_csv(export_file)
            assert len(exported_df) == 2
            assert "sequence_id" in exported_df.columns
            assert "score" in exported_df.columns

    def test_export_scores_nonexistent_model(self):
        """Test exporting scores for non-existent model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            export_file = os.path.join(tmpdir, "exported.csv")
            success = cache.export_scores("nonexistent_model", export_file)
            assert success is False

    def test_get_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")
            cache = ModelScoreCache(cache_file, load_community=False)

            # Empty cache stats
            stats = cache.get_cache_stats()
            assert stats["total_entries"] == 0
            assert stats["unique_models"] == 0
            assert stats["unique_sequences"] == 0

            # Add some data
            cache.add_scores(["seq1", "seq2"], [1.0, 2.0], "model1")
            cache.add_scores(["seq2", "seq3"], [2.5, 3.0], "model2")

            stats = cache.get_cache_stats()
            assert stats["total_entries"] == 4
            assert stats["unique_models"] == 2
            assert stats["unique_sequences"] == 3

    def test_persistence(self):
        """Test that cache persists to file and can be reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.csv")

            # Create cache and add data
            cache1 = ModelScoreCache(cache_file, load_community=False)
            cache1.add_scores(["seq1", "seq2"], [1.0, 2.0], "persistent_model")

            # Create new cache instance from same file
            cache2 = ModelScoreCache(cache_file, load_community=False)
            retrieved = cache2.get_scores(["seq1", "seq2"], "persistent_model")

            assert len(retrieved) == 2
            assert retrieved["seq1"] == 1.0
            assert retrieved["seq2"] == 2.0
