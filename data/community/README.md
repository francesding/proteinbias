# Community Model Contributions

This directory contains community-contributed protein bias benchmark results. Each model has its own subdirectory with all related files.

## Directory Structure

```
data/community/
├── README.md                    # This file
├── YourModelName/              # One directory per model
│   ├── model.py               # Model implementation
│   ├── scores.csv             # Pre-computed scores
│   └── summary.csv            # Benchmark results
├── AnotherModel/
│   ├── model.py
│   ├── scores.csv
│   └── summary.csv
└── ...
```

## Contributing Your Model

### Step 1: Create Your Model Directory

Create a new directory named after your model:
```bash
mkdir data/community/YourModelName
```

### Step 2: Implement Your Model

Create `model.py` with your model implementation:

```python
# data/community/YourModelName/model.py
from proteinbias import BaseModel

class YourModelName(BaseModel):
    def __init__(self):
        super().__init__("YourModelName")
        # Initialize your model here
    
    def score_sequence(self, sequence: str) -> float:
        # Your scoring logic here
        # Higher scores = better sequences
        return your_scoring_function(sequence)
    
    # Optional: for faster batch processing
    def score_sequences(self, sequences: List[str]) -> List[float]:
        return [self.score_sequence(seq) for seq in sequences]
```

### Step 3: Generate Scores and Results

Run the benchmark to generate your scores and summary:

```bash
# Generate scores and benchmark results
proteinbias run-benchmark \
    --model-file data/community/YourModelName/model.py \
    --model-class YourModelName \
    --output /tmp/results.csv \
    --summary data/community/YourModelName/summary.csv

# Export the scores to your model directory
proteinbias export-scores YourModelName data/community/YourModelName/scores.csv
```

### Step 4: Verify Your Contribution

Check that your model appears in the results (there needs to be `summary.csv` file in the expected place for this to work):
```bash
proteinbias list-results
```

## File Formats

### model.py
- Must inherit from `BaseModel`
- Must implement `score_sequence(sequence: str) -> float`
- Optionally implement `score_sequences()` for batch processing

### scores.csv
```csv
sequence_id,score,timestamp
P00001,1.23,2024-01-01T00:00:00
P00002,0.87,2024-01-01T00:00:00
...
```

### summary.csv
```csv
model,total_species,range,std_dev,iqr,eukaryota_mean,bacteria_mean,archaea_mean,mammalia_mean
YourModelName,138,245.8,62.3,89.2,1504.2,1478.1,1423.5,1512.7
```

## Viewing Results

```bash
# View all results (sorted alphabetically by default)
proteinbias list-results

# Sort by bias range (lower = less biased)
proteinbias list-results --sort-by range

# Sort by standard deviation
proteinbias list-results --sort-by std

# Sort by interquartile range
proteinbias list-results --sort-by iqr

# Sort by model name
proteinbias list-results --sort-by model_name
```

## Guidelines

### Model Naming
- Use descriptive, unique names
- Avoid spaces or special characters
- Examples: `MyModel_loglikelihood`, `MyModel_neg_energy`

### Reproducibility
- Add dependencies in an optional group in `pyproject.toml`
- Include model version/checkpoint information
- Note any special preprocessing requirements

## Automatic Integration
Models in this directory are automatically:
- **Loaded into cache**: Scores are available for `run-cached-benchmark`
- **Included in results**: Appear in `list-results` output
- **Discoverable**: Can be found and compared with other models
