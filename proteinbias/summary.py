"""Summary statistics for benchmark results."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BenchmarkSummarizer:
    """Generate summary statistics for benchmark results."""

    def __init__(
        self, results_df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None
    ):
        """Initialize the summarizer.

        Args:
            results_df: DataFrame with Elo benchmark results
            original_df: Original dataset with taxonomic information (optional)
        """
        self.results_df = results_df
        self.original_df = original_df
        self.species_taxonomy = self._extract_species_taxonomy()

    def _extract_species_taxonomy(self) -> Dict[str, Dict[str, str]]:
        """Extract taxonomic information for each species."""
        if self.original_df is None:
            return {}

        taxonomy = {}

        # Group by species and get first occurrence for taxonomy
        for genus_species, group in self.original_df.groupby("genus_species"):
            first_row = group.iloc[0]
            taxonomy[genus_species] = {
                "domain": first_row.get("domain", "Unknown"),
                "kingdom": first_row.get("kingdom", "Unknown"),
                "class": first_row.get("class", "Unknown"),
                "phylum_division": first_row.get("phylum_division", "Unknown"),
            }

        return taxonomy

    def _get_model_columns(self) -> List[str]:
        """Get all model columns from results DataFrame."""
        # Find columns that end with '_mean' (Elo ratings)
        mean_columns = [col for col in self.results_df.columns if col.endswith("_mean")]
        return mean_columns

    def _get_species_by_taxonomy(
        self, taxonomy_level: str, taxonomy_value: str
    ) -> List[str]:
        """Get species that belong to a specific taxonomic group."""
        if not self.species_taxonomy:
            return []

        species_list = []
        for species, taxonomy in self.species_taxonomy.items():
            tax_val = taxonomy.get(taxonomy_level, "")
            # Handle NaN values which are float objects
            if pd.isna(tax_val) or tax_val == "":
                continue
            if str(tax_val).lower() == taxonomy_value.lower():
                species_list.append(species)

        return species_list

    def _calculate_model_summary(self, model_column: str) -> Dict[str, Any]:
        """Calculate summary statistics for a single model."""
        # Filter out NaN values
        valid_data = self.results_df[self.results_df[model_column].notna()]

        if len(valid_data) == 0:
            return {
                "range": np.nan,
                "std_dev": np.nan,
                "iqr": np.nan,
                "eukaryota_mean": np.nan,
                "bacteria_mean": np.nan,
                "archaea_mean": np.nan,
                "mammalia_mean": np.nan,
                "total_species": 0,
            }

        values = valid_data[model_column]

        # Basic statistics
        range_val = values.max() - values.min()
        std_dev = values.std()
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        iqr = q75 - q25

        # Taxonomic group averages
        taxonomic_means = {}

        # Define taxonomic groups
        taxonomic_groups = {
            "eukaryota": ("domain", "Eukaryota"),
            "bacteria": ("domain", "Bacteria"),
            "archaea": ("domain", "Archaea"),
            "mammalia": ("class", "Mammalia"),
        }

        for group_name, (tax_level, tax_value) in taxonomic_groups.items():
            species_in_group = self._get_species_by_taxonomy(tax_level, tax_value)

            if species_in_group:
                # Filter results to species in this group
                group_data = valid_data[
                    valid_data["genus_species"].isin(species_in_group)
                ]
                if len(group_data) > 0:
                    taxonomic_means[f"{group_name}_mean"] = group_data[
                        model_column
                    ].mean()
                else:
                    taxonomic_means[f"{group_name}_mean"] = np.nan
            else:
                taxonomic_means[f"{group_name}_mean"] = np.nan

        summary = {
            "range": range_val,
            "std_dev": std_dev,
            "iqr": iqr,
            "total_species": len(valid_data),
            **taxonomic_means,
        }

        return summary

    def generate_summary(self) -> pd.DataFrame:
        """Generate complete summary statistics for all models.

        Returns:
            DataFrame with summary statistics for each model
        """
        model_columns = self._get_model_columns()

        if not model_columns:
            logger.warning("No model columns found in results DataFrame")
            return pd.DataFrame()

        summaries = []

        for model_col in model_columns:
            # Extract model name (remove '_mean' suffix)
            model_name = model_col.replace("_mean", "").replace("Elo_", "")

            summary = self._calculate_model_summary(model_col)
            summary["model"] = model_name
            summary["model_column"] = model_col

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)

        # Reorder columns for better readability
        column_order = [
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

        # Only include columns that exist
        existing_columns = [col for col in column_order if col in summary_df.columns]
        summary_df = summary_df[existing_columns]

        return summary_df

    def save_summary(self, output_file: str) -> None:
        """Save summary to CSV file.

        Args:
            output_file: Path to output CSV file
        """
        summary_df = self.generate_summary()
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Summary saved to: {output_file}")

    def get_taxonomic_breakdown(self) -> Dict[str, int]:
        """Get breakdown of species by taxonomic groups.

        Returns:
            Dictionary with counts for each taxonomic group
        """
        if not self.species_taxonomy:
            return {}

        breakdown = {
            "total_species": len(self.species_taxonomy),
            "eukaryota": 0,
            "bacteria": 0,
            "archaea": 0,
            "mammalia": 0,
            "unknown_domain": 0,
        }

        for species, taxonomy in self.species_taxonomy.items():
            domain = taxonomy.get("domain", "").lower()
            class_name = taxonomy.get("class", "").lower()

            if domain == "eukaryota":
                breakdown["eukaryota"] += 1
                if class_name == "mammalia":
                    breakdown["mammalia"] += 1
            elif domain == "bacteria":
                breakdown["bacteria"] += 1
            elif domain == "archaea":
                breakdown["archaea"] += 1
            else:
                breakdown["unknown_domain"] += 1

        return breakdown


def generate_benchmark_summary(
    results_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Generate benchmark summary statistics.

    Args:
        results_df: DataFrame with Elo benchmark results
        original_df: Original dataset with taxonomic information
        output_file: Path to save summary CSV (optional)

    Returns:
        DataFrame with summary statistics
    """
    summarizer = BenchmarkSummarizer(results_df, original_df)
    summary_df = summarizer.generate_summary()

    if output_file:
        summarizer.save_summary(output_file)

    return summary_df
