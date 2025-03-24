# Optimal Translation Quality Estimation Test

This directory contains an implementation of the translation quality estimation system with the optimal parameters identified in our research paper.

## Overview

The `optimal_test.py` script evaluates the performance of the quality estimation system using the parameters that our research identified as providing the best balance of sensitivity, precision, and computational efficiency.

## Optimal Parameters

Based on our extensive analysis documented in `research_paper.md`, we identified the following optimal parameters:

1. **TF-IDF Configuration**: 
   - `min_n=1, max_n=20`
   - This configuration provides the highest sensitivity to errors, particularly for multiple errors, while maintaining good detection rates for single errors.

2. **Sample Size**: 
   - `100 reference examples`
   - This sample size offers an optimal balance between detection accuracy and computational efficiency, with minimal gains observed from larger sample sizes.

3. **Quality Threshold**: 
   - `0.95`
   - This threshold value provides a good balance across different error counts, maintaining high recall while offering reasonable precision.

## Running the Test

Simply run the Python script directly:

```bash
python optimal_test.py
```

The script will automatically search for corpus files in common locations. If it can't find them, you'll need to adjust the paths in the script or call the function with the correct paths.

Required dependencies:
- numpy
- matplotlib
- pandas

## Output and Results

The script generates the following outputs in the `results/optimal_test/` directory:

1. **JSON Results**: 
   - Contains detailed data on downgrade rates and quality drops for different error types and counts.
   - The filename follows the pattern `optimal_test_results_TIMESTAMP.json`.

2. **Visualization**: 
   - A PNG image showing quality degradation vs. error count and a breakdown by error type.
   - The filename follows the pattern `optimal_test_visualization_TIMESTAMP.png`.

3. **Classification Metrics**: 
   - A CSV file with precision, recall, F1 scores, and accuracy at the optimal threshold.
   - A PNG visualization of these metrics.
   - Filenames follow the pattern `optimal_test_metrics_TIMESTAMP.csv` and `optimal_test_metrics_TIMESTAMP.png`.

## Interpreting Results

The results should confirm the findings from our research paper:

1. **Detection Rates**: 
   - Higher error counts should be detected with higher reliability.
   - Expect detection rates above 90% for most error types, especially with multiple errors.

2. **Quality Drops**: 
   - Quality drops should increase with error count.
   - Delete and swap errors typically show larger quality drops than insert and replace errors.

3. **Classification Performance**: 
   - F1 scores should increase with error count, reflecting better discrimination ability.
   - Recall should be consistently high (>90%) across all error levels.

## Advanced Usage

You can also import the `run_optimal_test` function directly in your own code for more control:

```python
from optimal_test import run_optimal_test

# Run with custom parameters
results = run_optimal_test(
    corpus_source_path="path/to/source.txt",
    corpus_target_path="path/to/target.txt",
    tfidf_min_n=1,
    tfidf_max_n=20,
    sample_size=100,
    quality_threshold=0.95,
    num_tests=40,
    max_errors=5
)
```

## Testing Alternative Configurations

To compare with other configurations discussed in the research paper, you can call the function with different parameter sets:

1. **Baseline Configuration**: 
   ```python
   run_optimal_test(tfidf_min_n=1, tfidf_max_n=3)
   ```

2. **Precision-Focused Configuration**: 
   ```python
   run_optimal_test(tfidf_min_n=3, tfidf_max_n=3)
   ```

3. **Small Sample Size Configuration**: 
   ```python
   run_optimal_test(sample_size=15)
   ```

## Further Development

This test implementation serves as a starting point for further refinement of the quality estimation system. Potential next steps include:

1. Testing with real-world translation errors rather than synthetic degradations.
2. Extending to word-level and semantic errors.
3. Evaluating with diverse language pairs.
4. Integrating with human evaluation for calibration. 