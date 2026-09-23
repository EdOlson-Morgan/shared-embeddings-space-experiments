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
│   ├── diversity.py          # Population diversity metrics (Vendi Score, etc.)
│   └── uuid_words/           # Token-efficient UUIDv4 <-> word encoding
├── scripts/
│   └── build_uuid_wordlist.py # Regenerates uuid_words/wordlist.py
├── tests/
│   └── test_uuid_words.py    # Run with `uv run pytest tests/`
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

## Utilities

### UUIDv4 Word Encoding
`src/embeddings_space/uuid_words/` reversibly encodes a UUIDv4 as a short
sequence of plain English words instead of raw hex, so it costs fewer LLM
tokens in prompts/transcripts while staying human-readable. See
[`docs/uuid-token-optimization-plan.md`](docs/uuid-token-optimization-plan.md)
for the design and measured results (the shipped wordlist is verified
single-token under `tiktoken`'s `o200k_base` encoding; the space-joined
form saves ~47% of tokens vs. a raw UUID string).

```python
from embeddings_space.uuid_words import encode_uuid4, decode_uuid4

encode_uuid4("173c6f57-2c5f-41a3-b898-f49eae04ffcc")
# "ability content shared prime queen think dude created fact frank bay wine"
```

## Voyage 4 Model Family

These models share a common embedding space, allowing for direct comparison and interoperability:
- `voyage-4-large`: Flagship model, maximum retrieval accuracy (MoE architecture).
- `voyage-4`: General purpose model, balanced performance.
- `voyage-4-lite`: Optimized for lower latency and reduced compute.
- `voyage-4-nano`: Lightweight, open-weight model for local development.
