"""Command-line interface for protein bias benchmarking."""

import argparse
import sys
import logging

from .benchmark import BenchmarkRunner
from .cache import ModelScoreCache
import os
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_run_baseline(args):
    """Run baseline benchmark."""
    try:
        runner = BenchmarkRunner()
        results = runner.run_baseline_benchmark(
            models=args.models,
            output_file=args.output,
            summary_file=args.summary,
            num_replicates=args.num_replicates,
            k_factor=args.k_factor,
            n_jobs=args.n_jobs,
        )

        print(f"Baseline benchmark completed. Results shape: {results.shape}")
        if args.output:
            print(f"Results saved to: {args.output}")
        if args.summary:
            print(f"Summary saved to: {args.summary}")
        if not args.output:
            print("\nTop results:")
            print(results.head())

    except Exception as e:
        print(f"Error running baseline benchmark: {e}")
        sys.exit(1)


def cmd_run_benchmark(args):
    """Run model benchmark."""
    try:
        cache = ModelScoreCache()
        runner = BenchmarkRunner(cache)

        # Load model from file if model_file provided
        if args.model_file:
            # Import the model file and get the model class
            import importlib.util

            spec = importlib.util.spec_from_file_location("user_model", args.model_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {args.model_file}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the model class
            if hasattr(module, args.model_class):
                model_cls = getattr(module, args.model_class)
                model = model_cls()
            else:
                raise ValueError(
                    f"Model class '{args.model_class}' not found in {args.model_file}"
                )

        results = runner.run_benchmark(
            model=model,
            output_file=args.output,
            summary_file=args.summary,
            num_replicates=args.num_replicates,
            k_factor=args.k_factor,
            n_jobs=args.n_jobs,
            use_cache=not args.no_use_cache,
            cache_scores=not args.no_cache,
        )

        print(f"Model benchmark completed. Results shape: {results.shape}")
        if args.output:
            print(f"Results saved to: {args.output}")
        if args.summary:
            print(f"Summary saved to: {args.summary}")
        if not args.output:
            print("\nTop results:")
            print(results.head())

    except Exception as e:
        print(f"Error running model benchmark: {e}")
        sys.exit(1)


def cmd_add_scores(args):
    """Add model scores from file to cache."""
    try:
        cache = ModelScoreCache()
        runner = BenchmarkRunner(cache)

        runner.add_scores_from_file(
            scores_file=args.scores_file,
            model_name=args.model_name,
            overwrite=args.overwrite,
        )

        print(f"Successfully added scores for {args.model_name}")

    except Exception as e:
        print(f"Error adding scores: {e}")
        sys.exit(1)


def cmd_list_scores(args):
    """List cached model scores."""
    try:
        cache = ModelScoreCache()
        models = cache.list_models()

        if not models:
            print("No cached model scores found.")
            return

        print("Cached model scores:")
        for model_info in models:
            print(f"  {model_info['model_name']}:")
            print(f"    Sequences: {model_info['num_sequences']}")
            print(
                f"    Score range: {model_info['score_range'][0]:.3f} to {model_info['score_range'][1]:.3f}"
            )
            print(f"    Latest: {model_info['latest_timestamp']}")
            if model_info["has_nan_scores"]:
                print("    Warning: Contains NaN scores")

        # Show overall cache stats
        stats = cache.get_cache_stats()
        print("\nCache statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Unique models: {stats['unique_models']}")
        print(f"  Unique sequences: {stats['unique_sequences']}")
        print(f"  Cache file size: {stats['cache_file_size_mb']:.2f} MB")

    except Exception as e:
        print(f"Error listing scores: {e}")
        sys.exit(1)


def cmd_export_scores(args):
    """Export cached scores for a model."""
    try:
        cache = ModelScoreCache()

        success = cache.export_scores(
            model_name=args.model_name, output_file=args.output, include_sequence=True
        )

        if success:
            print(f"Exported scores for {args.model_name} to {args.output}")
        else:
            print(f"No cached scores found for model: {args.model_name}")
            sys.exit(1)

    except Exception as e:
        print(f"Error exporting scores: {e}")
        sys.exit(1)


def cmd_clear_cache(args):
    """Clear cached scores."""
    try:
        cache = ModelScoreCache()

        if args.all:
            removed_count = cache.clear_all()
            print(f"Cleared all cache entries: {removed_count} removed")
        elif args.model_name:
            removed_count = cache.clear_model(args.model_name)
            if removed_count > 0:
                print(
                    f"Cleared cache for {args.model_name}: {removed_count} entries removed"
                )
            else:
                print(f"No cached entries found for model: {args.model_name}")
        else:
            print("Error: Must specify either --model-name or --all")
            sys.exit(1)

    except Exception as e:
        print(f"Error clearing cache: {e}")
        sys.exit(1)


def cmd_run_cached_benchmark(args):
    """Run benchmark using only cached scores."""
    try:
        cache = ModelScoreCache()
        runner = BenchmarkRunner(cache)

        model_names = args.models.split(",")

        results = runner.run_benchmark_from_cache(
            model_names=model_names,
            output_file=args.output,
            summary_file=args.summary,
            num_replicates=args.num_replicates,
            k_factor=args.k_factor,
            n_jobs=args.n_jobs,
        )

        print(f"Cached benchmark completed. Results shape: {results.shape}")
        if args.output:
            print(f"Results saved to: {args.output}")
        if args.summary:
            print(f"Summary saved to: {args.summary}")
        if not args.output:
            print("\nTop results:")
            print(results.head())

    except Exception as e:
        print(f"Error running cached benchmark: {e}")
        sys.exit(1)


def cmd_list_results(args):
    """List bias results for all models from results files."""
    try:
        # Look for result files in community model directories
        package_dir = os.path.dirname(os.path.dirname(__file__))
        community_dir = os.path.join(package_dir, "data/community")
        baseline_results_file = os.path.join(package_dir, "baseline_summary.csv")

        # Start with baseline results if available
        all_results = []
        if os.path.exists(baseline_results_file):
            try:
                baseline_df = pd.read_csv(baseline_results_file)
                if not baseline_df.empty:
                    all_results.append(baseline_df)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # Skip empty or malformed baseline file
                pass

        # Add community results from model subdirectories
        if os.path.exists(community_dir):
            for item in os.listdir(community_dir):
                model_dir = os.path.join(community_dir, item)

                # Skip files and non-directories
                if not os.path.isdir(model_dir) or item.startswith("."):
                    continue

                # Skip README and other documentation directories
                if item.lower() in ["readme", "docs", "examples"]:
                    continue

                # Look for summary.csv in this model directory
                summary_file = os.path.join(model_dir, "summary.csv")
                if os.path.exists(summary_file):
                    try:
                        df = pd.read_csv(summary_file)
                        if not df.empty:
                            all_results.append(df)
                    except Exception as e:
                        print(f"Warning: Could not read {summary_file}: {e}")

        if not all_results:
            print("No benchmark results found.")
            return

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Remove duplicates by keeping only one row per unique model
        # Priority: baseline file > community files (to ensure consistent baseline results)
        deduplicated_df = combined_df.drop_duplicates(subset=["model"], keep="first")

        # Sort results if requested (no ranking implied)
        if args.sort_by == "range":
            sorted_df = deduplicated_df.sort_values("range", ascending=True)
        elif args.sort_by == "std":
            sorted_df = deduplicated_df.sort_values("std_dev", ascending=True)
        elif args.sort_by == "iqr":
            sorted_df = deduplicated_df.sort_values("iqr", ascending=True)
        elif args.sort_by == "model_name":
            sorted_df = deduplicated_df.sort_values("model", ascending=True)
        else:  # none
            sorted_df = deduplicated_df

        # Output results
        if args.output:
            sorted_df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        else:
            print("\nModel Bias Results:")
            if args.sort_by != "none":
                print(f"(sorted by {args.sort_by})")
            print("=" * 100)
            print(sorted_df.to_string(index=False))

    except Exception as e:
        print(f"Error listing results: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Protein bias benchmarking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run baseline benchmark command
    baseline_parser = subparsers.add_parser(
        "run-baseline", help="Run baseline benchmark"
    )
    baseline_parser.add_argument(
        "--models", nargs="+", help="Specific models to benchmark"
    )
    baseline_parser.add_argument(
        "--output", default="baseline_results.csv", help="Output CSV file path"
    )
    baseline_parser.add_argument(
        "--summary",
        default="baseline_summary.csv",
        help="Summary CSV file path",
    )
    baseline_parser.add_argument(
        "--num-replicates",
        type=int,
        default=50,
        help="Number of replicates (default: 50)",
    )
    baseline_parser.add_argument(
        "--k-factor", type=int, default=16, help="Elo K-factor (default: 16)"
    )
    baseline_parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs")

    # Run model benchmark command
    benchmark_parser = subparsers.add_parser(
        "run-benchmark", help="Run model benchmark"
    )
    benchmark_parser.add_argument(
        "--model-file", help="Path to Python file containing model"
    )
    benchmark_parser.add_argument(
        "--model-class", default="MyModel", help="Model class name (default: MyModel)"
    )
    benchmark_parser.add_argument(
        "--output", default="results.csv", help="Output CSV file path"
    )
    benchmark_parser.add_argument(
        "--summary", default="summary.csv", help="Summary CSV file path"
    )
    benchmark_parser.add_argument(
        "--num-replicates",
        type=int,
        default=50,
        help="Number of replicates (default: 50)",
    )
    benchmark_parser.add_argument(
        "--k-factor", type=int, default=16, help="Elo K-factor (default: 16)"
    )
    benchmark_parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs")
    benchmark_parser.add_argument(
        "--no-cache", action="store_true", help="Disable score caching"
    )
    benchmark_parser.add_argument(
        "--no-use-cache", action="store_true", help="Don't use cached scores"
    )

    # Add scores command
    add_scores_parser = subparsers.add_parser(
        "add-scores", help="Add model scores from file to cache"
    )
    add_scores_parser.add_argument("scores_file", help="Path to CSV file with scores")
    add_scores_parser.add_argument("model_name", help="Name of the model")
    add_scores_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing scores"
    )

    # List cached scores command
    subparsers.add_parser("list-scores", help="List cached model scores")

    # Export scores command
    export_scores_parser = subparsers.add_parser(
        "export-scores", help="Export cached scores for a model"
    )
    export_scores_parser.add_argument("model_name", help="Name of the model")
    export_scores_parser.add_argument("output", help="Output CSV file path")

    # Clear cache command
    clear_cache_parser = subparsers.add_parser(
        "clear-cache", help="Clear cached scores"
    )
    clear_cache_group = clear_cache_parser.add_mutually_exclusive_group(required=True)
    clear_cache_group.add_argument(
        "--model-name", help="Clear scores for specific model"
    )
    clear_cache_group.add_argument(
        "--all", action="store_true", help="Clear all cached scores"
    )

    # Run cached benchmark command
    cached_benchmark_parser = subparsers.add_parser(
        "run-cached-benchmark", help="Run benchmark using only cached scores"
    )
    cached_benchmark_parser.add_argument(
        "models", help="Comma-separated list of model names"
    )
    cached_benchmark_parser.add_argument(
        "--output", default="results.csv", help="Output CSV file path"
    )
    cached_benchmark_parser.add_argument(
        "--summary", default="summary.csv", help="Summary CSV file path"
    )
    cached_benchmark_parser.add_argument(
        "--num-replicates",
        type=int,
        default=50,
        help="Number of replicates (default: 50)",
    )
    cached_benchmark_parser.add_argument(
        "--k-factor", type=int, default=16, help="Elo K-factor (default: 16)"
    )
    cached_benchmark_parser.add_argument(
        "--n-jobs", type=int, help="Number of parallel jobs"
    )

    # List results command
    list_results_parser = subparsers.add_parser(
        "list-results", help="List bias results for all models"
    )
    list_results_parser.add_argument(
        "--sort-by",
        choices=["range", "std", "iqr", "model_name", "none"],
        default="none",
        help="Sort results by model (default: none - no ranking implied)",
    )
    list_results_parser.add_argument(
        "--output", help="Output file for results (optional)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == "run-baseline":
        cmd_run_baseline(args)
    elif args.command == "run-benchmark":
        # Validate model arguments
        if not args.model_file:
            print("Error: --model-file must be provided")
            sys.exit(1)
        cmd_run_benchmark(args)
    elif args.command == "add-scores":
        cmd_add_scores(args)
    elif args.command == "list-scores":
        cmd_list_scores(args)
    elif args.command == "export-scores":
        cmd_export_scores(args)
    elif args.command == "clear-cache":
        cmd_clear_cache(args)
    elif args.command == "run-cached-benchmark":
        cmd_run_cached_benchmark(args)
    elif args.command == "list-results":
        cmd_list_results(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
