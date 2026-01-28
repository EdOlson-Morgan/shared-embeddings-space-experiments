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
│   ├── metrics.py            # Similarity metrics
│   └── diversity.py          # Population diversity metrics (Vendi Score, etc.)
├── data/                     # Sample datasets
│   ├── paraphrases.json      # Multi-topic paraphrase groups
│   └── coffee_paraphrases.json # Homogeneous single-topic dataset
├── notebooks/                # Experiment notebooks
│   ├── 01_paraphrase_similarity.ipynb
│   ├── 02_population_diversity.ipynb
│   └── 03_cross_model_comparison.ipynb
└── .env.example              # Environment template
```

## Experiments

### 01 - Paraphrase Similarity
Explores how Voyage embeddings capture semantic similarity between individual paraphrases versus unrelated text using standard metrics like cosine similarity.

### 02 - Embedding Population Diversity
Uses advanced metrics like the **Vendi Score** and **Effective Rank** to quantitatively compare the diversity of a heterogeneous population (multiple topics) against a homogeneous one (single topic).

### 03 - Cross-Model Embedding Comparison
Analyzes the consistency across the Voyage 4 family within their shared embedding space. Compares how large, standard, lite, and nano models represent the same content and whether they agree on similarity rankings.

## Voyage 4 Model Family

These models share a common embedding space, allowing for direct comparison and interoperability:
- `voyage-4-large`: Flagship model, maximum retrieval accuracy (MoE architecture).
- `voyage-4`: General purpose model, balanced performance.
- `voyage-4-lite`: Optimized for lower latency and reduced compute.
- `voyage-4-nano`: Lightweight, open-weight model for local development.
