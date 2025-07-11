# ProteinBias: Benchmarking Species Bias in Protein Models

**A standardized framework to evaluate and quantify species bias in protein sequence models.**

## What is Species Bias?

Many protein sequence models exhibit **species bias** – they systematically assign higher scores (e.g., likelihoods) to sequences from certain species (often mammals) and lower scores to others (often Archaea), even for the same protein. This can affect the interpretation of these scores across the tree of life, and the bias can be detrimental for some protein design applications.

## What This Tool Does

ProteinBias uses an **Elo rating system** to quantify species bias in any protein scoring model:

- ✅ **Easy benchmarking**: Compare your model against established baselines with one command
- ✅ **Quantitative bias measurement**: Get precise Elo ratings showing which species your model favors  
- ✅ **Reproducible results**: Validate your setup by reproducing published baseline results

> **Paper**: This framework accompanies our paper: [*Protein language models are biased by unequal sequence sampling across the tree of life*](https://www.biorxiv.org/content/10.1101/2024.03.07.584001v1)

## How It Works

ProteinBias measures bias by comparing how your model scores **orthologous proteins** (same protein function, different species):

1. **Load protein sequences** from diverse species that share the same protein name in UniProt
2. **Score sequences** using your model (or use our baseline models)  
3. **Run Elo tournaments** where species "compete" based on their protein sequence scores
4. **Analyze Elo rating results** to see which species received high Elo ratings (i.e., systematically higher model scores)
5. **Get bias metrics** to compare models on their Elo rating distributions

**Large Elo rating spread = more bias in the model**, with the caveat that bias in protein scoring models can sometimes be desirable, depending on the application (see paper for discussion).

> **Technical details**: Species start at Elo 1500, then "compete" in pairwise matchups based on model scores. Winners gain points, losers lose points using the [standard Elo algorithm](https://en.wikipedia.org/wiki/Elo_rating_system). Results are averaged across multiple random tournament orderings for stability.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd proteinbias

# Install with uv (recommended)
uv sync
source .venv/bin/activate
uv pip install -e .[dev]

# Or install with pip
pip install -e .[dev]
```

## Quick Start

### 1. First, validate your setup

```bash
# Test with existing baseline models (ProGen2, ESM2 variants)
proteinbias run-baseline --output baseline_results.csv --summary baseline_summary.csv
```

This ensures everything works by reproducing published results. Compare the outputs to `data/core/baseline_results.csv` and `data/core/baseline_summary.csv`.

### 2. Benchmark your own model

**Step 1:** Create your model class

```python
# my_model.py
from proteinbias import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super().__init__("MyModel")
    
    def score_sequence(self, sequence: str) -> float:
        # Your scoring logic here (higher = better)
        return your_scoring_function(sequence)
```

**Step 2:** Run the benchmark

```bash
proteinbias run-benchmark \
    --model-file my_model.py \
    --model-class MyModel \
    --output results.csv \
    --summary summary.csv
```

### 3. Interpret your results

The `results.csv` shows **Elo ratings for each species** and the standard error of the mean Elo rating across replicates:

| genus_species | Elo_Progen2_xlarge_ll_mean | Elo_Progen2_xlarge_ll_SE | Elo_MyModel_mean | Elo_MyModel_SE |
|---------------|----------------------------| -------------------------|------------------|------------------|
| *Homo sapiens*  | 2405.3    | 12.5  | 1547.3 | 8.0
| *Halobacterium salinarum* | 1113.3 | 27.7  | 1492.1 | 6.5

**Key insights:**
- **1500 = neutral rating** (no bias)
- **Higher ratings** = your model favors this species
- **Large spread** across species = more bias in your model

The `summary.csv` shows aggregated bias metrics to compare models at a high level:

| model | range | std_dev | iqr | eukaryota_mean | bacteria_mean | archaea_mean | mammalia_mean |
|-------|-------|---------|-----|-----------|----------|---------|----------|
| Progen2_xlarge_ll | 2040.0 | 563.2 | 612.3 | 1562.9 | 1523.1 | 846.0 | 2148.2 |
| MyModel | 345.8 | 62.3 | 89.2 | 1504.2 | 1478.1 | 1423.5 | 1512.7 |

**In this example, MyModel seems to show less bias than ProGen2** (smaller range and IQR, lower standard deviation). Note that lower bias is not necessarily always desirable -- see the paper for discussion of interpreting bias.


---

## Advanced Usage

### Score Caching

Scores are automatically cached and reused, so if a model is slow at scoring protein sequences, you only need to run scoring once:

```bash
# First run: computes and caches scores, then runs Elo computation
proteinbias run-benchmark --model-file my_model.py --model-class MyModel

# Second run: uses cached scores (instant! then runs Elo computation)
proteinbias run-benchmark --model-file my_model.py --model-class MyModel
```

**Alternative workflows:**

You can also compute scores outside of this benchmark and add these scores to the cache:
```bash
# Add pre-computed scores from file
proteinbias add-scores my_scores.csv "MyModel"

# Run benchmark using only cached scores (multiple models)
proteinbias run-cached-benchmark "Model1,Model2,Model3" --output results.csv

# List cached models
proteinbias list-scores

# Export scores for a model
proteinbias export-scores MyModel my_model_scores.csv
```

**Score file format:**
sequence_id | score
----------- | -----
P12345      | 1.23
Q67890      | 0.87

Note: The cache system uses sequence IDs (from the UniProt `Entry` column of the default dataset) for efficient lookup. 


### Batch Processing

If your model supports batch processing, override the `score_sequences` method:

```python
class MyBatchModel(BaseModel):
    def score_sequences(self, sequences: List[str]) -> List[float]:
        # Implement efficient batch scoring
        return batch_score(sequences)
```

### Configuration Options

```bash
# Run with custom parameters
proteinbias run-benchmark \
    --model-file my_model.py \
    --num-replicates 100 \
    --k-factor 32 \
    --n-jobs 8 \
    --output results.csv \
    --summary summary.csv
```

## Model Implementation Examples

See `examples/example_model.py` for several example implementations:

- **RandomModel**: Simple random baseline
- **SequenceLengthModel**: Length-based scoring
- **AminoAcidCompositionModel**: Composition-based scoring

## Default Dataset

The default dataset is the one used for analysis in our paper, comprised of two files:
- **Protein Sequences**: `data/core/curated_protein_sequences.csv` - sequences and metadata
- **Baseline Scores**: `data/core/baseline_scores.csv` - ProGen2 and ESM2 model scores for sequences in the above file

Additionally, data used for the "time to the last common ancestor" analysis in the paper can be found at `data/core/selected_species_with_time_to_common_ancestor_matrix.csv`.

Currently the benchmark only supports the default dataset, but contributions of new datasets would be welcome!

## Community Contributions

**Share your model results with the research community!**

Community models are organized in individual directories under `data/community/`. Each model gets its own subdirectory containing:
- `model.py` - Model implementation
- `scores.csv` - Pre-computed scores for the benchmark dataset
- `summary.csv` - Benchmark results and bias statistics

See [`data/community/README.md`](data/community/README.md) for detailed contribution guidelines.

```bash
# View all model results (baseline + community)
proteinbias list-results
```

**Important**: The `list-results` command only shows:
1. **Baseline models** (from the default benchmark dataset)
2. **Community models** (from `data/community/` subdirectories)

It combines the `summary.csv` files for different models, and does not depend on the raw scores. To see your own model in `list-results`, you must add your benchmark summary to the community directory as described in the contribution guidelines above.

## CLI Reference

<details>
<summary><strong>All Commands</strong> (click to expand)</summary>

**Benchmarking:**
```bash
proteinbias run-baseline [--output] [--summary] [--num-replicates] [--k-factor]
proteinbias run-benchmark [--model-file] [--model-class] [--output] [--summary]
proteinbias run-cached-benchmark <models> [--output] [--summary]
```

**Score Management:**
```bash
proteinbias add-scores <file> <model_name> [--overwrite]
proteinbias list-scores
proteinbias export-scores <model_name> <output_file>
proteinbias clear-cache [--model-name <name> | --all]
```


**Model Results:**
```bash
proteinbias list-results [--sort-by range|std|iqr|model_name|none]
```
</details>

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install dev requirements and pre-commit tool
```bash
uvx pre-commit install
```
3. Add your improvements
4. Submit a pull request
5. Thank you!

## License

MIT License - see LICENSE file for details.

## Citation

If you use this benchmarking framework in your research, please cite:

```bibtex
@article{ding2024protein,
  title={Protein language models are biased by unequal sequence sampling across the tree of life},
  author={Ding, Frances and Steinhardt, Jacob},
  journal={BioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
