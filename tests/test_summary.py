"""Tests for summary statistics functionality."""

import pytest
import pandas as pd
import numpy as np
from proteinbias.summary import BenchmarkSummarizer, generate_benchmark_summary


class TestBenchmarkSummarizer:
    """Test the BenchmarkSummarizer class."""

    @pytest.fixture
    def sample_results_df(self):
        """Create sample results DataFrame."""
        return pd.DataFrame(
            {
                "genus_species": [
                    "Species_A",
                    "Species_B",
                    "Species_C",
                    "Species_D",
                    "Species_E",
                ],
                "Elo_TestModel_mean": [1400, 1500, 1600, 1700, 1800],
                "Elo_AnotherModel_mean": [1200, 1400, 1600, 1800, 2000],
            }
        )

    @pytest.fixture
    def sample_original_df(self):
        """Create sample original DataFrame with taxonomy."""
        return pd.DataFrame(
            {
                "genus_species": [
                    "Species_A",
                    "Species_B",
                    "Species_C",
                    "Species_D",
                    "Species_E",
                ],
                "domain": ["Bacteria", "Bacteria", "Eukaryota", "Eukaryota", "Archaea"],
                "class": ["Unknown", "Unknown", "Mammalia", "Unknown", "Unknown"],
            }
        )

    def test_initialization(self, sample_results_df, sample_original_df):
        """Test summarizer initialization."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        assert summarizer.results_df is sample_results_df
        assert summarizer.original_df is sample_original_df
        assert len(summarizer.species_taxonomy) == 5

    def test_initialization_without_taxonomy(self, sample_results_df):
        """Test summarizer initialization without taxonomy data."""
        summarizer = BenchmarkSummarizer(sample_results_df)
        assert summarizer.results_df is sample_results_df
        assert summarizer.original_df is None
        assert summarizer.species_taxonomy == {}

    def test_extract_species_taxonomy(self, sample_results_df, sample_original_df):
        """Test species taxonomy extraction."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        taxonomy = summarizer.species_taxonomy

        assert "Species_A" in taxonomy
        assert taxonomy["Species_A"]["domain"] == "Bacteria"
        assert taxonomy["Species_C"]["class"] == "Mammalia"
        assert taxonomy["Species_E"]["domain"] == "Archaea"

    def test_get_model_columns(self, sample_results_df, sample_original_df):
        """Test model column detection."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        model_columns = summarizer._get_model_columns()

        assert "Elo_TestModel_mean" in model_columns
        assert "Elo_AnotherModel_mean" in model_columns
        assert "genus_species" not in model_columns

    def test_calculate_model_summary_basic_stats(
        self, sample_results_df, sample_original_df
    ):
        """Test basic statistics calculation."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        summary = summarizer._calculate_model_summary("Elo_TestModel_mean")

        # Test basic statistics
        assert summary["total_species"] == 5
        assert summary["range"] == 400  # 1800 - 1400
        assert abs(summary["std_dev"] - 158.11) < 0.1  # Standard deviation
        assert summary["iqr"] == 200  # Q75(1700) - Q25(1500)

    def test_calculate_model_summary_taxonomic_groups(
        self, sample_results_df, sample_original_df
    ):
        """Test taxonomic group calculations."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        summary = summarizer._calculate_model_summary("Elo_TestModel_mean")

        # Test taxonomic means
        assert summary["bacteria_mean"] == 1450  # (1400 + 1500) / 2
        assert summary["eukaryota_mean"] == 1650  # (1600 + 1700) / 2
        assert summary["archaea_mean"] == 1800  # Single value
        assert summary["mammalia_mean"] == 1600  # Single species in class

    def test_calculate_model_summary_empty_data(self, sample_original_df):
        """Test handling of empty data."""
        empty_df = pd.DataFrame({"genus_species": [], "Elo_TestModel_mean": []})
        summarizer = BenchmarkSummarizer(empty_df, sample_original_df)
        summary = summarizer._calculate_model_summary("Elo_TestModel_mean")

        assert summary["total_species"] == 0
        assert pd.isna(summary["range"])
        assert pd.isna(summary["std_dev"])
        assert pd.isna(summary["iqr"])

    def test_calculate_model_summary_nan_values(self, sample_original_df):
        """Test handling of NaN values."""
        nan_df = pd.DataFrame(
            {
                "genus_species": ["Species_A", "Species_B", "Species_C"],
                "Elo_TestModel_mean": [1400, np.nan, 1600],
            }
        )
        summarizer = BenchmarkSummarizer(nan_df, sample_original_df)
        summary = summarizer._calculate_model_summary("Elo_TestModel_mean")

        assert summary["total_species"] == 2  # Excludes NaN
        assert summary["range"] == 200  # 1600 - 1400

    def test_generate_summary(self, sample_results_df, sample_original_df):
        """Test complete summary generation."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        summary_df = summarizer.generate_summary()

        assert len(summary_df) == 2  # Two models
        assert "model" in summary_df.columns
        assert "range" in summary_df.columns
        assert "std_dev" in summary_df.columns
        assert "iqr" in summary_df.columns
        assert "total_species" in summary_df.columns

        # Check model names are extracted correctly
        models = summary_df["model"].tolist()
        assert "TestModel" in models
        assert "AnotherModel" in models

    def test_generate_summary_column_order(self, sample_results_df, sample_original_df):
        """Test that summary columns are in expected order."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        summary_df = summarizer.generate_summary()

        expected_order = [
            "model",
            "total_species",
            "range",
            "std_dev",
            "iqr",
            "eukaryota_mean",
            "bacteria_mean",
            "archaea_mean",
            "mammalia_mean",
        ]

        assert list(summary_df.columns) == expected_order

    def test_get_taxonomic_breakdown(self, sample_results_df, sample_original_df):
        """Test taxonomic breakdown calculation."""
        summarizer = BenchmarkSummarizer(sample_results_df, sample_original_df)
        breakdown = summarizer.get_taxonomic_breakdown()

        assert breakdown["total_species"] == 5
        assert breakdown["bacteria"] == 2
        assert breakdown["eukaryota"] == 2
        assert breakdown["archaea"] == 1
        assert breakdown["mammalia"] == 1

    def test_get_taxonomic_breakdown_no_taxonomy(self, sample_results_df):
        """Test taxonomic breakdown without taxonomy data."""
        summarizer = BenchmarkSummarizer(sample_results_df)
        breakdown = summarizer.get_taxonomic_breakdown()

        assert breakdown == {}


class TestGenerateBenchmarkSummary:
    """Test the standalone generate_benchmark_summary function."""

    def test_generate_benchmark_summary_function(self, tmp_path):
        """Test the convenience function."""
        results_df = pd.DataFrame(
            {
                "genus_species": ["Species_A", "Species_B"],
                "Elo_TestModel_mean": [1400, 1600],
            }
        )

        output_file = tmp_path / "test_summary.csv"

        summary_df = generate_benchmark_summary(
            results_df, output_file=str(output_file)
        )

        assert len(summary_df) == 1
        assert summary_df.iloc[0]["model"] == "TestModel"
        assert output_file.exists()

    def test_generate_benchmark_summary_without_file(self):
        """Test function without saving to file."""
        results_df = pd.DataFrame(
            {
                "genus_species": ["Species_A", "Species_B"],
                "Elo_TestModel_mean": [1400, 1600],
            }
        )

        summary_df = generate_benchmark_summary(results_df)

        assert len(summary_df) == 1
        assert summary_df.iloc[0]["model"] == "TestModel"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_model_columns(self):
        """Test with DataFrame containing no model columns."""
        df = pd.DataFrame(
            {"genus_species": ["Species_A", "Species_B"], "some_other_column": [1, 2]}
        )

        summarizer = BenchmarkSummarizer(df)
        summary_df = summarizer.generate_summary()

        assert len(summary_df) == 0

    def test_missing_genus_species_column(self):
        """Test with DataFrame missing genus_species column."""
        df = pd.DataFrame(
            {"species": ["Species_A", "Species_B"], "Elo_TestModel_mean": [1400, 1600]}
        )

        # Should not crash, but taxonomic calculations will fail gracefully
        summarizer = BenchmarkSummarizer(df)
        summary_df = summarizer.generate_summary()

        assert len(summary_df) == 1  # Still processes the model
