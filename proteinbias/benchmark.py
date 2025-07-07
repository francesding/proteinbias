"""Benchmark registry and execution system."""

import os
import pandas as pd
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Union

from .models import BaseModel
from .compute_elo import get_replicate_ratings_parallel, consolidate_replicate_ratings
from .summary import generate_benchmark_summary
from .cache import ModelScoreCache

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Simple dataset configuration provider."""

    @staticmethod
    def get_dataset() -> Dict[str, Any]:
        """Get the default dataset configuration."""
        return {
            "name": "Common SwissProt Proteins Dataset",
            "description": "Dataset of proteins with many orthologous sequences across representative species",
            "sequences_path": "data/core/curated_protein_sequences.csv",
            "baseline_scores_path": "data/core/baseline_scores.csv",
            "baseline_models": [
                "Progen2_medium_ll",
                "Progen2_base_ll",
                "Progen2_large_ll",
                "Progen2_xlarge_ll",
                "Progen2_BFD90_ll",
                "ESM2_650M_pppl",
                "ESM2_3B_pppl",
                "ESM2_15B_pppl",
            ],
        }


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, cache: Optional[ModelScoreCache] = None):
        """Initialize the benchmark runner.

        Args:
            cache: ModelScoreCache instance. If None, creates a default one.
        """
        self.cache = cache or ModelScoreCache()

    def _load_model_from_file(self, model_path: str, model_class: str) -> BaseModel:
        """Load a model class from a Python file.

        Args:
            model_path: Path to the Python file
            model_class: Full class name

        Returns:
            Instantiated model
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location("user_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {model_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class
        if "." in model_class:
            # Handle nested class names
            class_parts = model_class.split(".")
            cls = module
            for part in class_parts:
                cls = getattr(cls, part)
        else:
            cls = getattr(module, model_class)

        # Instantiate the model
        return cls()

    def _load_dataset(self) -> pd.DataFrame:
        """Load the default dataset using split data structure.

        Returns:
            DataFrame with merged sequences and baseline scores
        """
        dataset_config = DatasetConfig.get_dataset()

        # Get paths for split data files
        sequences_path = dataset_config["sequences_path"]
        baseline_scores_path = dataset_config["baseline_scores_path"]

        # Make paths relative to package if not absolute
        package_dir = os.path.dirname(os.path.dirname(__file__))

        if not os.path.isabs(sequences_path):
            sequences_path = os.path.join(package_dir, sequences_path)
        if not os.path.isabs(baseline_scores_path):
            baseline_scores_path = os.path.join(package_dir, baseline_scores_path)

        # Load both files
        if not os.path.exists(sequences_path):
            raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
        if not os.path.exists(baseline_scores_path):
            raise FileNotFoundError(
                f"Baseline scores file not found: {baseline_scores_path}"
            )

        sequences_df = pd.read_csv(sequences_path)
        scores_df = pd.read_csv(baseline_scores_path)

        # Merge on Entry column
        merged_df = sequences_df.merge(scores_df, on="Entry", how="inner")
        logger.info(f"Loaded dataset: {len(merged_df)} sequences")

        return merged_df

    def _get_model_scores(
        self, model: BaseModel, df: pd.DataFrame, use_cache: bool, cache_scores: bool
    ) -> List[float]:
        """Get model scores using cache when possible.

        Args:
            model: Model instance
            df: DataFrame containing sequences and sequence IDs
            use_cache: Whether to use cached scores
            cache_scores: Whether to cache new scores

        Returns:
            List of scores for the sequences
        """
        sequences = df["sequence"].tolist()
        sequence_ids = df["Entry"].tolist()

        scores = [None] * len(sequences)
        sequences_to_score = []
        sequence_ids_to_score = []
        indices_to_score = []

        # Check cache if enabled
        if use_cache:
            cached_scores = self.cache.get_scores(sequence_ids, model.name)
            cache_hits = 0

            for i, (seq, seq_id) in enumerate(zip(sequences, sequence_ids)):
                if seq_id in cached_scores:
                    scores[i] = cached_scores[seq_id]
                    cache_hits += 1
                else:
                    sequences_to_score.append(seq)
                    sequence_ids_to_score.append(seq_id)
                    indices_to_score.append(i)

            if cache_hits > 0:
                logger.info(
                    f"Cache hit: {cache_hits}/{len(sequences)} scores for {model.name}"
                )
        else:
            sequences_to_score = sequences
            sequence_ids_to_score = sequence_ids
            indices_to_score = list(range(len(sequences)))

        # Score remaining sequences
        if sequences_to_score:
            logger.info(
                f"Computing {len(sequences_to_score)} new scores with {model.name}"
            )
            new_scores = model.score_sequences(sequences_to_score)

            # Fill in the new scores
            for idx, score in zip(indices_to_score, new_scores):
                scores[idx] = score

            # Cache new scores if enabled
            if cache_scores:
                try:
                    self.cache.add_scores(sequence_ids_to_score, new_scores, model.name)
                except Exception as e:
                    logger.warning(f"Failed to cache scores for {model.name}: {e}")

        return scores

    def _run_elo_benchmark(
        self,
        df: pd.DataFrame,
        models: List[str],
        num_replicates: int,
        k_factor: int,
        n_jobs: Optional[int],
    ) -> pd.DataFrame:
        """Helper method to run Elo benchmark and consolidate results."""
        logger.info(f"Running Elo benchmark with {num_replicates} replicates")
        ratings_replicates = get_replicate_ratings_parallel(
            df,
            models=models,
            num_replicates=num_replicates,
            k=k_factor,
            n_jobs=n_jobs,
        )
        return consolidate_replicate_ratings(ratings_replicates, models)

    def _save_results_and_summary(
        self,
        results_df: pd.DataFrame,
        df: pd.DataFrame,
        output_file: Optional[str],
        summary_file: Optional[str],
    ):
        """Helper method to save results and generate summary."""
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")

        if summary_file:
            generate_benchmark_summary(results_df, df, summary_file)

    def run_benchmark_from_cache(
        self,
        model_names: List[str],
        output_file: Optional[str] = "results.csv",
        summary_file: Optional[str] = "summary.csv",
        num_replicates: int = 50,
        k_factor: int = 16,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run benchmark using only cached model scores.

        Args:
            model_names: List of model names to include in benchmark
            output_file: Path to save results CSV (optional)
            summary_file: Path to save summary statistics CSV (optional)
            num_replicates: Number of Elo replicates to run
            k_factor: K-factor for Elo calculations
            n_jobs: Number of parallel jobs

        Returns:
            DataFrame with benchmark results
        """
        # Load dataset
        dataset_config = DatasetConfig.get_dataset()
        df = self._load_dataset()
        sequence_ids = df["Entry"].tolist()

        # Get scores for all models from cache
        all_models = dataset_config["baseline_models"].copy()

        for model_name in model_names:
            cached_scores = self.cache.get_scores(sequence_ids, model_name)

            if len(cached_scores) != len(sequence_ids):
                missing_count = len(sequence_ids) - len(cached_scores)
                raise ValueError(
                    f"Model {model_name} missing scores for {missing_count} sequences. "
                    f"Run model to generate all scores first."
                )

            # Add scores to dataframe
            model_column = model_name
            scores = [
                cached_scores.get(seq_id, float("nan")) for seq_id in sequence_ids
            ]
            df[model_column] = scores
            all_models.append(model_column)

        logger.info(f"Running benchmark with {len(model_names)} cached models")

        # Run Elo benchmark
        results_df = self._run_elo_benchmark(
            df, all_models, num_replicates, k_factor, n_jobs
        )

        # Save results and summary
        self._save_results_and_summary(results_df, df, output_file, summary_file)

        return results_df

    def add_scores_from_file(
        self,
        scores_file: str,
        model_name: str,
        overwrite: bool = False,
    ) -> bool:
        """Add model scores from file to cache.

        Args:
            scores_file: Path to CSV file with scores
            model_name: Name of the model
            overwrite: Whether to overwrite existing scores

        Returns:
            True if successful
        """
        # Load dataset for validation
        df = self._load_dataset()
        dataset_sequence_ids = set(df["Entry"].tolist())

        # Add scores to cache
        added_count = self.cache.add_scores_from_file(
            scores_file, model_name, overwrite
        )

        # Validate that all dataset sequences are covered
        cached_scores = self.cache.get_scores(list(dataset_sequence_ids), model_name)
        missing_count = len(dataset_sequence_ids) - len(cached_scores)

        if missing_count > 0:
            logger.warning(
                f"Model {model_name} missing scores for {missing_count} dataset sequences"
            )

        logger.info(f"Added {added_count} scores for {model_name} from {scores_file}")
        return True

    def run_baseline_benchmark(
        self,
        models: Optional[List[str]] = None,
        output_file: Optional[str] = "baseline_results.csv",
        summary_file: Optional[str] = "baseline_summary.csv",
        num_replicates: int = 50,
        k_factor: int = 16,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run benchmark on baseline metrics only (no new model).

        This is useful for:
        - Validating the benchmark setup
        - Reproducing published baseline results
        - Testing different benchmark parameters

        Args:
            models: List of baseline models to include. If None, uses all baseline models.
            output_file: Path to save results CSV (optional)
            summary_file: Path to save summary statistics CSV (optional)
            num_replicates: Number of Elo replicates to run
            k_factor: K-factor for Elo calculations
            n_jobs: Number of parallel jobs

        Returns:
            DataFrame with benchmark results
        """
        # Load dataset
        dataset_config = DatasetConfig.get_dataset()
        df = self._load_dataset()

        # Use specified models or all baseline models
        if models is None:
            models = dataset_config["baseline_models"]

        # Validate that all models exist in the dataset
        missing_models = [m for m in models if m not in df.columns]
        if missing_models:
            raise ValueError(f"Models not found in dataset: {missing_models}")

        logger.info("Running baseline benchmark")
        logger.info(f"Models: {models}")

        # Run Elo benchmark
        results_df = self._run_elo_benchmark(
            df, models, num_replicates, k_factor, n_jobs
        )

        # Save results and summary
        self._save_results_and_summary(results_df, df, output_file, summary_file)

        return results_df

    def run_benchmark(
        self,
        model: Union[BaseModel, str],
        output_file: Optional[str] = "results.csv",
        summary_file: Optional[str] = "summary.csv",
        num_replicates: int = 50,
        k_factor: int = 16,
        n_jobs: Optional[int] = None,
        use_cache: bool = True,
        cache_scores: bool = True,
    ) -> pd.DataFrame:
        """Run a benchmark comparing a model against baselines.

        Args:
            model: Either a BaseModel instance or registered model name
            output_file: Path to save results CSV (optional)
            summary_file: Path to save summary statistics CSV (optional)
            num_replicates: Number of Elo replicates to run
            k_factor: K-factor for Elo calculations
            n_jobs: Number of parallel jobs
            use_cache: Whether to use cached scores if available
            cache_scores: Whether to cache computed scores for future use

        Returns:
            DataFrame with benchmark results
        """
        # Load model if string provided
        # Model should always be a BaseModel instance now
        if isinstance(model, str):
            raise ValueError(
                "String model names are no longer supported. Use BaseModel instances."
            )

        # Load dataset
        dataset_config = DatasetConfig.get_dataset()
        df = self._load_dataset()

        # Score sequences with the new model (using cache if available)
        scores = self._get_model_scores(model, df, use_cache, cache_scores)

        # Add new model scores to dataframe
        model_column = model.name
        df[model_column] = scores

        # Run Elo benchmark including the new model
        all_models = dataset_config["baseline_models"] + [model_column]

        # Run Elo benchmark
        results_df = self._run_elo_benchmark(
            df, all_models, num_replicates, k_factor, n_jobs
        )

        # Save results and summary
        self._save_results_and_summary(results_df, df, output_file, summary_file)

        return results_df
