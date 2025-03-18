# PairQuery: Translation Benchmark 

This module provides tools for two main purposes:
1. **Benchmarking Translation Performance**: Compare zero-shot vs. few-shot performance of language models on translation tasks
2. **Generating Training Data**: Create high-quality, context-aware examples for fine-tuning translation models

## Components

### 1. Translation Benchmarking (translator.py)

The benchmarking system allows you to compare LLM translation performance with and without examples:

Key features:
- Compare zero-shot vs. few-shot translation performance
- Test with different numbers of examples (e.g., 0, 5, 10)
- Generate comprehensive visualizations including:
  - Progressive performance charts showing how scores evolve as more samples are tested
  - Comparative bar charts showing performance across different example counts
  - Line charts displaying the effect of example count on translation quality
  - Smoothed trend lines for better visualization
- Calculate similarity using multiple metrics (sequence similarity, word overlap, word order)
- Save detailed results in JSON format for further analysis

### 2. Parallel Corpus Handler (extract.py)

The `PairCorpus` class manages parallel text in two languages:

Key features:
- Reads and manages source and target language files
- Provides similarity search to find related translations
- Formats prompts for translation with or without examples
- Supports benchmarking by generating test cases

### 3. Training Data Generation

For fine-tuning translation models, the system can generate training data:

## Usage Examples

### Running a Translation Benchmark

To compare zero-shot and few-shot translation performance:

```python
from pair_query.translator import run_benchmark

results = run_benchmark(
    corpus_path_source="ebible/corpus/eng-engULB.txt", 
    corpus_path_target="ebible/corpus/kos-kos.txt", 
    num_examples=[0, 5, 10],  # Compare zero-shot (0) with 5 and 10 examples
    num_tests=20,  # Number of test samples
    language="Kosraean"  # Target language name
)

# The results contain detailed information about performance
print(f"Zero-shot avg. similarity: {results[0]['avg_similarity']:.4f}")
print(f"5-shot avg. similarity: {results[5]['avg_similarity']:.4f}")
print(f"10-shot avg. similarity: {results[10]['avg_similarity']:.4f}")
```
