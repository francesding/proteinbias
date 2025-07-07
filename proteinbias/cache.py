"""Model score cache system for reusable benchmarking."""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ModelScoreCache:
    """Manages cached model scores for reusable benchmarking."""

    def __init__(self, cache_file: Optional[str] = None, load_community: bool = True):
        """Initialize the model score cache.

        Args:
            cache_file: Path to cache CSV file. If None, uses default location.
            load_community: Whether to automatically load community score files.
        """
        if cache_file is None:
            cache_file = os.path.join(
                os.path.dirname(__file__), "..", "model_scores_cache.csv"
            )

        self.cache_file = cache_file
        self.load_community = load_community
        self.cache_df = self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        """Load existing cache and optionally incorporate community score files."""
        # Load existing cache file
        cache_df = pd.DataFrame(
            columns=["sequence_id", "model_name", "score", "timestamp"]
        )

        if os.path.exists(self.cache_file):
            try:
                cache_df = pd.read_csv(self.cache_file)
                logger.info(
                    f"Loaded cache with {len(cache_df)} entries from {self.cache_file}"
                )
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning(
                    f"Error loading cache file: {e}. Starting with empty cache."
                )

        # Load community score files and integrate into cache (if enabled)
        if self.load_community:
            community_scores = self._load_community_scores()
            if not community_scores.empty:
                # Merge community scores, keeping cache entries when there are conflicts
                cache_df = self._merge_community_scores(cache_df, community_scores)

        return cache_df

    def _load_community_scores(self) -> pd.DataFrame:
        """Load all community score files from model subdirectories."""
        # Find community directory
        package_dir = os.path.dirname(os.path.dirname(__file__))
        community_dir = os.path.join(package_dir, "data/community")

        if not os.path.exists(community_dir):
            return pd.DataFrame(
                columns=["sequence_id", "model_name", "score", "timestamp"]
            )

        all_community_scores = []

        # Scan for model subdirectories
        for item in os.listdir(community_dir):
            model_dir = os.path.join(community_dir, item)

            # Skip files and non-directories
            if not os.path.isdir(model_dir) or item.startswith("."):
                continue

            # Skip README and other documentation directories
            if item.lower() in ["readme", "docs", "examples"]:
                continue

            # Look for scores.csv in this model directory
            scores_file = os.path.join(model_dir, "scores.csv")
            if os.path.exists(scores_file):
                try:
                    model_name = item  # Use directory name as model name

                    # Load scores
                    scores_df = pd.read_csv(scores_file)

                    # Validate format
                    if (
                        "sequence_id" not in scores_df.columns
                        or "score" not in scores_df.columns
                    ):
                        if "Entry" in scores_df.columns:
                            scores_df = scores_df.rename(
                                columns={"Entry": "sequence_id"}
                            )
                        else:
                            logger.warning(
                                f"Community score file {scores_file} missing required columns"
                            )
                            continue

                    # Add model name and timestamp
                    scores_df["model_name"] = model_name
                    scores_df["timestamp"] = f"community_model_{model_name}"

                    # Keep only required columns
                    scores_df = scores_df[
                        ["sequence_id", "model_name", "score", "timestamp"]
                    ]
                    all_community_scores.append(scores_df)

                    logger.info(
                        f"Loaded {len(scores_df)} community scores for {model_name}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Error loading community score file {scores_file}: {e}"
                    )

        if all_community_scores:
            return pd.concat(all_community_scores, ignore_index=True)
        else:
            return pd.DataFrame(
                columns=["sequence_id", "model_name", "score", "timestamp"]
            )

    def _merge_community_scores(
        self, cache_df: pd.DataFrame, community_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge community scores into cache, keeping existing cache entries when there are conflicts."""
        if community_df.empty:
            return cache_df

        if cache_df.empty:
            logger.info(
                f"Included {len(community_df)} community scores in addition to cache"
            )
            return community_df

        # Find community scores that don't conflict with cache
        cache_keys = (
            cache_df[["sequence_id", "model_name"]].apply(tuple, axis=1).tolist()
        )
        community_keys = community_df[["sequence_id", "model_name"]].apply(
            tuple, axis=1
        )

        # Keep community scores that aren't already in cache
        new_community_mask = ~community_keys.isin(cache_keys)
        new_community_scores = community_df[new_community_mask]

        if not new_community_scores.empty:
            merged_df = pd.concat([cache_df, new_community_scores], ignore_index=True)
            logger.info(
                f"Included {len(new_community_scores)} community scores in addition to cache"
            )
            return merged_df
        else:
            logger.info("No new community scores to add (all already in cache)")
            return cache_df

    def _save_cache(self):
        """Save cache to file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.cache_df.to_csv(self.cache_file, index=False)
        logger.info(f"Cache saved to {self.cache_file}")

    def _validate_inputs(
        self, sequence_ids: List[str], scores: List[float], model_name: str
    ) -> bool:
        """Validate that sequence IDs and scores are properly formatted."""
        if len(sequence_ids) != len(scores):
            raise ValueError(
                f"Number of sequence IDs ({len(sequence_ids)}) must match number of scores ({len(scores)})"
            )

        if not sequence_ids:
            raise ValueError("No sequence IDs provided")

        if not model_name.strip():
            raise ValueError("Model name cannot be empty")

        # Check for valid scores
        for i, score in enumerate(scores):
            if pd.isna(score):
                logger.warning(
                    f"NaN score found for sequence ID {sequence_ids[i]}, model {model_name}"
                )

        return True

    def add_scores(
        self,
        sequence_ids: List[str],
        scores: List[float],
        model_name: str,
        overwrite: bool = False,
    ) -> int:
        """Add model scores to cache.

        Args:
            sequence_ids: List of sequence IDs
            scores: List of corresponding scores
            model_name: Name of the model
            overwrite: If True, overwrite existing scores for same model-sequence pairs

        Returns:
            Number of new entries added to cache
        """
        self._validate_inputs(sequence_ids, scores, model_name)

        new_entries = []
        existing_count = 0

        for sequence_id, score in zip(sequence_ids, scores):
            # Check if this model-sequence_id combination already exists
            existing_mask = (self.cache_df["sequence_id"] == sequence_id) & (
                self.cache_df["model_name"] == model_name
            )

            if existing_mask.any():
                if overwrite:
                    # Remove existing entry
                    self.cache_df = self.cache_df[~existing_mask]
                    logger.debug(
                        f"Overwriting existing score for {model_name}, sequence ID {sequence_id}"
                    )
                else:
                    existing_count += 1
                    continue

            new_entries.append(
                {
                    "sequence_id": sequence_id,
                    "model_name": model_name,
                    "score": score,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
            )

        if new_entries:
            new_df = pd.DataFrame(new_entries)
            if self.cache_df.empty:
                self.cache_df = new_df
            else:
                self.cache_df = pd.concat([self.cache_df, new_df], ignore_index=True)
            self._save_cache()

        if existing_count > 0:
            logger.info(f"Skipped {existing_count} existing entries for {model_name}")

        logger.info(f"Added {len(new_entries)} new scores for {model_name}")
        return len(new_entries)

    def get_scores(self, sequence_ids: List[str], model_name: str) -> Dict[str, float]:
        """Get cached scores for sequence IDs.

        Args:
            sequence_ids: List of sequence IDs
            model_name: Name of the model

        Returns:
            Dictionary mapping sequence IDs to scores (only for found sequence IDs)
        """
        # Filter cache for this model
        model_cache = self.cache_df[self.cache_df["model_name"] == model_name]

        # Create mapping from sequence ID to score
        score_map = {}
        for _, row in model_cache.iterrows():
            sequence_id = row["sequence_id"]
            if sequence_id in sequence_ids:
                score_map[sequence_id] = row["score"]

        return score_map

    def has_model_scores(
        self, sequence_ids: List[str], model_name: str
    ) -> Dict[str, bool]:
        """Check which sequence IDs have cached scores for a model.

        Args:
            sequence_ids: List of sequence IDs
            model_name: Name of the model

        Returns:
            Dictionary mapping sequence IDs to whether they have cached scores
        """
        cached_scores = self.get_scores(sequence_ids, model_name)
        return {seq_id: seq_id in cached_scores for seq_id in sequence_ids}

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in cache with statistics.

        Returns:
            List of dictionaries with model information
        """
        if len(self.cache_df) == 0:
            return []

        model_stats = []
        for model_name in self.cache_df["model_name"].unique():
            model_data = self.cache_df[self.cache_df["model_name"] == model_name]

            stats = {
                "model_name": model_name,
                "num_sequences": len(model_data),
                "latest_timestamp": model_data["timestamp"].max(),
                "score_range": (model_data["score"].min(), model_data["score"].max()),
                "has_nan_scores": model_data["score"].isna().any(),
            }
            model_stats.append(stats)

        return model_stats

    def export_scores(
        self, model_name: str, output_file: str, include_sequence: bool = True
    ) -> bool:
        """Export scores for a model to CSV file.

        Args:
            model_name: Name of the model
            output_file: Path to output CSV file
            include_sequence: Whether to include sequence column (deprecated, always includes sequence_id)

        Returns:
            True if export successful, False if model not found
        """
        model_data = self.cache_df[self.cache_df["model_name"] == model_name]

        if len(model_data) == 0:
            logger.warning(f"No cached scores found for model: {model_name}")
            return False

        # Select columns for export
        export_columns = ["sequence_id", "score", "timestamp"]

        export_df = model_data[export_columns].copy()
        export_df.to_csv(output_file, index=False)

        logger.info(
            f"Exported {len(export_df)} scores for {model_name} to {output_file}"
        )
        return True

    def clear_model(self, model_name: str) -> int:
        """Remove all scores for a specific model.

        Args:
            model_name: Name of the model to remove

        Returns:
            Number of entries removed
        """
        initial_size = len(self.cache_df)
        self.cache_df = self.cache_df[self.cache_df["model_name"] != model_name]
        removed_count = initial_size - len(self.cache_df)

        if removed_count > 0:
            self._save_cache()
            logger.info(f"Removed {removed_count} entries for model: {model_name}")

        return removed_count

    def clear_all(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries removed
        """
        removed_count = len(self.cache_df)
        self.cache_df = pd.DataFrame(
            columns=["sequence_id", "model_name", "score", "timestamp"]
        )
        self._save_cache()

        logger.info(f"Cleared all cache entries: {removed_count} removed")
        return removed_count

    def add_scores_from_file(
        self, scores_file: str, model_name: str, overwrite: bool = False
    ) -> int:
        """Add scores from CSV file to cache.

        Args:
            scores_file: Path to CSV file with scores
            model_name: Name of the model
            overwrite: Whether to overwrite existing scores

        Returns:
            Number of new entries added
        """
        try:
            scores_df = pd.read_csv(scores_file)
        except Exception as e:
            raise ValueError(f"Error reading scores file {scores_file}: {e}")

        # Validate file format - support both old (sequence) and new (sequence_id) formats
        if "sequence_id" in scores_df.columns:
            sequence_ids = scores_df["sequence_id"].tolist()
        elif "sequence" in scores_df.columns:
            sequence_ids = scores_df["sequence"].tolist()
        else:
            raise ValueError(
                "Scores file must contain 'sequence_id' or 'sequence' column"
            )

        if "score" not in scores_df.columns:
            raise ValueError("Scores file must contain 'score' column")

        scores = scores_df["score"].tolist()

        return self.add_scores(sequence_ids, scores, model_name, overwrite)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get overall cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if len(self.cache_df) == 0:
            return {
                "total_entries": 0,
                "unique_models": 0,
                "unique_sequences": 0,
                "cache_file_size_mb": 0.0,
            }

        stats = {
            "total_entries": len(self.cache_df),
            "unique_models": self.cache_df["model_name"].nunique(),
            "unique_sequences": self.cache_df["sequence_id"].nunique(),
            "cache_file_size_mb": os.path.getsize(self.cache_file) / (1024 * 1024)
            if os.path.exists(self.cache_file)
            else 0.0,
        }

        return stats
