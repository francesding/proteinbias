import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random
from itertools import combinations

import numpy as np
import pandas as pd
from numba import jit

MODELS = [
    "Progen2_xlarge_ll",
    "Progen2_medium_ll",
    "Progen2_base_ll",
    "Progen2_large_ll",
    "Progen2_BFD90_ll",
    "ESM2_650M_pppl",
    "ESM2_3B_pppl",
    "ESM2_15B_pppl",
]


# Optimized Elo calculation functions with numba JIT compilation
@jit(nopython=True)
def expected_outcome(rating1: float, rating2: float) -> float:
    """
    Calculate the expected outcome of a match between two players based on their Elo ratings.

    Args:
        rating1 (float): Elo rating of player 1
        rating2 (float): Elo rating of player 2

    Returns:
        float: Expected probability that player 1 wins (between 0 and 1)
    """
    return 1.0 / (1.0 + math.pow(10.0, (rating2 - rating1) / 400.0))


@jit(nopython=True)
def update_elo(
    player_rating: float, opponent_rating: float, result: float, k: int = 32
) -> float:
    """
    Update a player's Elo rating based on the outcome of a match.

    Args:
        player_rating (float): Current Elo rating of the player
        opponent_rating (float): Elo rating of the opponent
        result (float): Actual result (1.0 for win, 0.0 for loss, 0.5 for draw)
        k (int, optional): K-factor that determines how much ratings change. Defaults to 32.

    Returns:
        float: New Elo rating for the player
    """
    expected = expected_outcome(player_rating, opponent_rating)
    new_rating = player_rating + k * (result - expected)
    return new_rating


class EloBenchmarker:
    """
    Memory-efficient Elo rating computation for species-based model benchmarking.

    This class avoids creating large matchup DataFrames by sampling matchups
    on-the-fly and updating Elo scores iteratively across multiple epochs.
    """

    def __init__(self, df, models=MODELS, initial_rating=1500.0, k=16):
        """
        Initialize the Elo benchmarker.

        Args:
            df (pd.DataFrame): Input DataFrame containing protein sequences and models
            models (list): List of model column names to process
            initial_rating (float): Initial Elo rating for new species
            k (int): K-factor for Elo rating updates
        """
        self.df = df
        self.models = models
        self.initial_rating = initial_rating
        self.k = k

        # Pre-compute species medians for each protein and metric
        self._compute_species_medians()

        # Initialize Elo ratings
        self.elo_ratings = {model: {} for model in models}

    def _compute_species_medians(self):
        """Pre-compute median scores for each species-model-protein combination."""
        self.species_medians = {}

        for protein in self.df.first_protein_name.unique():
            sub_df = self.df[self.df.first_protein_name == protein]
            for model in self.models:
                medians = sub_df.groupby("genus_species")[model].median()
                for species, median_val in medians.items():
                    self.species_medians[(protein, species, model)] = median_val

    def _get_species_pairs(self, protein):
        """Get all possible species pairs for a given protein."""
        species_list = self.df[
            self.df.first_protein_name == protein
        ].genus_species.unique()
        return list(combinations(species_list, 2))

    def _compute_matchup_result(self, protein, species_1, species_2, model):
        """Compute the result of a matchup between two species for a given model."""
        score_1 = self.species_medians.get((protein, species_1, model), 0.0)
        score_2 = self.species_medians.get((protein, species_2, model), 0.0)

        if score_1 > score_2:
            return 1.0
        elif score_1 < score_2:
            return 0.0
        else:
            return 0.5

    def _initialize_ratings_if_needed(self, species_1, species_2, model):
        """Initialize Elo ratings for species if they don't exist."""
        if species_1 not in self.elo_ratings[model]:
            self.elo_ratings[model][species_1] = self.initial_rating
        if species_2 not in self.elo_ratings[model]:
            self.elo_ratings[model][species_2] = self.initial_rating

    def _update_ratings_for_matchup(self, protein, species_1, species_2, model):
        """Update Elo ratings for a single matchup."""
        result = self._compute_matchup_result(protein, species_1, species_2, model)

        self._initialize_ratings_if_needed(species_1, species_2, model)

        player1_rating = self.elo_ratings[model][species_1]
        player2_rating = self.elo_ratings[model][species_2]

        # Update Elo ratings for both players
        new_player1_rating = update_elo(player1_rating, player2_rating, result, self.k)
        new_player2_rating = update_elo(
            player2_rating, player1_rating, 1.0 - result, self.k
        )

        self.elo_ratings[model][species_1] = new_player1_rating
        self.elo_ratings[model][species_2] = new_player2_rating

    def compute_elo_single_epoch(self, random_state=None):
        """
        Compute Elo ratings for one epoch.

        Args:
            random_state (int, optional): Random seed for reproducible sampling

        Returns:
            dict: Final Elo ratings after this epoch
        """
        if random_state is not None:
            random.seed(random_state)

        # Get all proteins
        proteins = self.df.first_protein_name.unique()
        # Shuffle proteins for this epoch
        random.shuffle(proteins)

        # For each protein, sample species pairs and update ratings
        for protein in proteins:
            species_pairs = self._get_species_pairs(protein)

            # Shuffle pairs for this epoch
            random.shuffle(species_pairs)

            for species_1, species_2 in species_pairs:
                for model in self.models:
                    self._update_ratings_for_matchup(
                        protein, species_1, species_2, model
                    )

        return self.elo_ratings.copy()

    def get_ratings_dataframe(self):
        """Convert current Elo ratings to a DataFrame."""
        data = []
        for model in self.models:
            for species, rating in self.elo_ratings[model].items():
                data.append(
                    {"genus_species": species, "model": model, "elo_rating": rating}
                )

        return pd.DataFrame(data)


def process_single_replicate(args):
    """
    Worker function for parallel processing of replicates using the new approach.

    Args:
        args (tuple): Tuple containing (df, models, k, random_state)

    Returns:
        dict: Dictionary containing Elo ratings for this replicate
    """
    df, models, k, random_state = args

    benchmarker = EloBenchmarker(df, models=models, k=k)
    return benchmarker.compute_elo_single_epoch(random_state=random_state)


def get_replicate_ratings_parallel(
    df, models=MODELS, num_replicates=50, k=16, n_jobs=None
):
    """
    Compute Elo ratings across multiple replicates using parallel processing.

    Args:
        df (pd.DataFrame): Input DataFrame containing protein sequences and models
        models (list, optional): List of model columns to process. Defaults to MODELS.
        num_replicates (int, optional): Number of replicates to run. Defaults to 50.
        k (int, optional): K-factor for Elo rating updates. Defaults to 32.
        n_jobs (int, optional): Number of parallel jobs. Defaults to None (uses CPU count).

    Returns:
        list: List of dictionaries, each containing Elo ratings for one replicate
    """
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), num_replicates)

    # Prepare arguments for parallel processing
    args_list = [(df, models, k, idx) for idx in range(num_replicates)]

    ratings_replicates = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_replicate, args): idx
            for idx, args in enumerate(args_list)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                ratings_replicates.append(result)
            except Exception as exc:
                print(
                    f"Replicate {future_to_idx[future]} generated an exception: {exc}"
                )

    return ratings_replicates


def consolidate_replicate_ratings(ratings_replicates, models):
    """
    Consolidate Elo ratings from multiple replicates into mean and std statistics.

    Args:
        ratings_replicates (list): List of dictionaries, each containing Elo ratings for one replicate
        models (list): List of model names that were processed

    Returns:
        pd.DataFrame: DataFrame with species as rows and mean/std Elo ratings as columns
    """
    # Collect all species across all replicates and models
    all_species = set()
    for replicate_ratings in ratings_replicates:
        for model in models:
            all_species.update(replicate_ratings[model].keys())

    # Initialize results dictionary
    results = {}
    for species in all_species:
        results[species] = {}
        for model in models:
            results[species][model] = []

    # Collect all ratings for each species-model combination
    for replicate_ratings in ratings_replicates:
        for model in models:
            for species, rating in replicate_ratings[model].items():
                results[species][model].append(rating)

    # Compute statistics and create DataFrame
    data = []
    for species in all_species:
        row = {"genus_species": species}
        for model in models:
            ratings = results[species][model]
            if ratings:  # Only add if we have ratings for this species-model
                row[f"Elo_{model}_mean"] = np.mean(ratings)
                row[f"Elo_{model}_SE"] = np.std(ratings) / np.sqrt(len(ratings))
        data.append(row)

    return pd.DataFrame(data)


def main():
    """
    Main function to compute Elo ratings from sequence likelihood data.

    This function reads sequence likelihood data, creates species matchups,
    computes Elo ratings across multiple replicates, and saves the results.

    Command line arguments:
        --seq_ll_csv: Path to input CSV file with sequence likelihood data
        --model: Model column name to process
        --out_replicate_csv: Path to output CSV file for replicate results
        --n_jobs: Number of parallel jobs (default: number of CPU cores)
        --num_replicates: Number of replicates (default: 50)

    Returns:
        pd.DataFrame: DataFrame containing mean and standard error (SE) of Elo ratings for each species.
                      The SE is computed across replicates.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_ll_csv", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--out_replicate_csv", type=str)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: number of CPU cores)",
    )
    parser.add_argument(
        "--num_replicates",
        type=int,
        default=50,
        help="Number of replicates (default: 50)",
    )
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.seq_ll_csv)

    # Use the new optimized approach
    ratings_replicates = get_replicate_ratings_parallel(
        df,
        models=[args.model],
        num_replicates=args.num_replicates,
        n_jobs=args.n_jobs,
    )

    # Consolidate results directly to mean/std statistics
    ratings_df = consolidate_replicate_ratings(ratings_replicates, models=[args.model])

    if args.out_replicate_csv:
        ratings_df.to_csv(args.out_replicate_csv, index=False)

    return ratings_df


if __name__ == "__main__":
    main()
