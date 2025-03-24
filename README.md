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

# Translation Quality Error Sensitivity Experiments

This repository contains tools to analyze how the Quality Estimator responds to errors in translations. The experiments help determine optimal quality thresholds and understand the system's sensitivity to different types of errors.

## Overview

The error sensitivity testing framework allows you to:

1. **Analyze how the Quality Estimator responds to errors** - We introduce controlled errors into translations to see how well the system can detect them.
2. **Find optimal quality thresholds** - Determine the best quality score cutoffs to separate good translations from those containing errors.
3. **Test different parameters** - Evaluate how sample size and TF-IDF parameters affect quality estimation.

## Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Quick Start

To run a quick test with default settings:

```bash
python tests/run_error_experiments.py --baseline --samples 10 --quick
```

This will run a basic experiment with 10 samples and save results in the `results/error_experiments` directory.

### Experiment Types

You can run different types of experiments:

```bash
# Run all experiments (this can take a long time with many samples)
python tests/run_error_experiments.py --all

# Run just the baseline experiment
python tests/run_error_experiments.py --baseline

# Find optimal quality thresholds
python tests/run_error_experiments.py --thresholds

# Test different sample sizes
python tests/run_error_experiments.py --sample-size

# Test different TF-IDF parameters
python tests/run_error_experiments.py --tfidf-params
```

### Common Options

```
--samples N           Number of samples to test (default: 10)
--error-level {word,character}
                      Level at which to introduce errors (default: character)
--max-errors N        Maximum number of errors to introduce (default: 5)
--examples N          Number of examples for quality estimation (default: 20)
--min-n N             Default min_n for TF-IDF (default: 1)
--max-n N             Default max_n for TF-IDF (default: 3)
--quick               Run with reduced parameters for quicker testing
```

## Understanding the Results

The experiments generate visualizations and JSON data in the `results/error_experiments` directory.

### Key Visualizations

1. **Baseline Experiment** (`error_sensitivity_base.png`):
   - Shows how the percentage of translations with quality downgrade changes with the number of errors
   - Shows the average quality drop for different error types and counts

2. **Threshold Analysis** (`threshold_optimization_thresholds.png`):
   - Shows optimal quality thresholds for detecting different numbers of errors
   - Includes accuracy metrics to help you choose the right threshold
   - The explanation panel helps interpret what the chart means

3. **Precision-Recall Curves** (`threshold_optimization_precision_recall.png`):
   - Shows how well the system can detect errors at different sensitivity levels
   - Higher curves indicate better error detection

### Interpreting Quality Thresholds

The threshold chart shows:
- **Quality thresholds (blue line)**: Translations with scores BELOW this value should be flagged for review
- **Accuracy (green line)**: The percentage of translations that will be correctly classified at each threshold

For practical use:
- Use the **single error threshold** when you need to catch even small errors
- Use the **multiple error threshold** when you're more concerned with significant errors

## Running in Production

For production use, we recommend:
1. Run the threshold experiments with at least 100 samples
2. Use the determined threshold in your quality estimation pipeline
3. Periodically re-run the experiments as your corpus grows

## Troubleshooting

If you encounter issues:
- Ensure your corpus files exist at the specified paths
- Try running with fewer samples using the `--quick` flag
- Check the log output for specific error messages
