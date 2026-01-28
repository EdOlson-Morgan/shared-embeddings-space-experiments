# Shared Embeddings Space Experiments

Experiments exploring shared embedding spaces, focusing on Voyage AI's Voyage 4 series models.

## Setup

1. **Install dependencies** (using [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```

2. **Configure API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Voyage AI API key
   ```

3. **Run Jupyter notebooks**:
   ```bash
   uv run jupyter notebook
   ```

## Project Structure

```
├── src/embeddings_space/     # Core modules
│   ├── embeddings.py         # Voyage AI client wrapper
│   └── metrics.py            # Similarity metrics
├── data/                     # Sample datasets
│   └── paraphrases.json      # Paraphrase examples
├── notebooks/                # Experiment notebooks
│   └── 01_paraphrase_similarity.ipynb
└── .env.example              # Environment template
```

## Experiments

### 01 - Paraphrase Similarity
Explores how Voyage embeddings capture semantic similarity between paraphrases of the same content.

## Voyage 4 Model Family

These models share a common embedding space:
- `voyage-3.5` - General purpose
- `voyage-3.5-lite` - Faster/lighter version
- `voyage-code-3` - Code optimized
- `voyage-finance-2` - Finance domain
- `voyage-law-2` - Legal domain
