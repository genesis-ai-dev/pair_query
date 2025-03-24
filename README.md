# PairQuery: Translation Quality Estimation and Benchmarking

## Translation Quality Estimation

PairQuery provides a novel approach to evaluate translation quality without requiring ground truth references. The method works by analyzing correlation patterns between source and target languages:

1. **Similarity Correlation Method**: 
   - For a given source text and its translation, we sample N examples from a parallel corpus
   - We measure how similar the source text is to each example's source text (e.g., [0.3, 0.1, 0.2])
   - We measure how similar the translation is to each example's target text (e.g., [0.4, 0.2, 0.3])
   - A high correlation between these similarity patterns indicates a good translation

2. **Multiple Similarity Measures**:
   - N-gram similarity (sequence of words)
   - Word overlap (shared vocabulary)
   - TF-IDF cosine similarity (weighted term importance)
   - Word edit distance (Levenshtein distance)
   - Length similarity (comparing text proportions)

3. **Combination Strategies**:
   - Multiply correlations (joint probability approach)
   - Weighted average of correlations

This approach allows for quality estimation in low-resource languages where traditional reference-based metrics may not be applicable.

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

For fine-tuning translation models, the system can generate training data.

## Usage Examples

### Running a Translation Benchmark

To compare zero-shot and few-shot translation performance:
