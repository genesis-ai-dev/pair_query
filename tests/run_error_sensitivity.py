#!/usr/bin/env python3
"""
Script to analyze how the Quality Estimator responds to multiple errors.
This script tests error counts from 1 to 5 for each error type and creates a simple visualization.
It can also compare different example selection methods (random vs search-based vs random-then-sort).
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import multiprocessing
import argparse
from typing import List

# Add the parent directory to the Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get the parent directory path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from error_sensitivity import analyze_multiple_error_sensitivity, ErrorType, visualize_multiple_error_results
from pq.main import PairCorpus, QualityEstimator
from pq.similarity_measures import (
    NGramSimilarity,
    WordOverlapSimilarity,
    TfidfCosineSimilarity
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values
SOURCE_PATH = os.path.join(PARENT_DIR, "files/corpus/eng-engULB.txt")
TARGET_PATH = os.path.join(PARENT_DIR, "files/corpus/kos-kos.txt")
ERROR_LEVEL = "character"  # Options: "word" or "character"
NUM_SAMPLES = 30      # Number of samples to test
MAX_ERRORS = 5        # Maximum number of errors to introduce
EXAMPLE_SIZE = 20     # Number of examples for quality estimation
# Automatically determine the number of CPUs for parallelization
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "error_sensitivity_results.png")  # Output file for visualization
COMPARE_SELECTION = False  # Whether to compare random vs search-based selection
RANDOM_POOL_SIZE = 1000    # Default size of random pool for 'random_then_sort' method

def create_summary_visualization(results, error_level):
    """Create a clear, informative visualization of the error sensitivity results."""
    # This function is now replaced by the visualize_multiple_error_results function
    # from error_sensitivity module, which handles both single results and comparisons
    return visualize_multiple_error_results(results, error_level)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run error sensitivity analysis')
    parser.add_argument('--source', default=SOURCE_PATH, help='Path to source corpus file')
    parser.add_argument('--target', default=TARGET_PATH, help='Path to target corpus file')
    parser.add_argument('--level', default=ERROR_LEVEL, choices=['word', 'character'], 
                        help='Level at which to introduce errors')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES, help='Number of samples to test')
    parser.add_argument('--max-errors', type=int, default=MAX_ERRORS, help='Maximum number of errors to introduce')
    parser.add_argument('--example-size', type=int, default=EXAMPLE_SIZE, 
                        help='Number of examples for quality estimation')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Number of worker processes')
    parser.add_argument('--output-dir', default=RESULTS_DIR, help='Directory to save results')
    parser.add_argument('--compare', action='store_true', default=COMPARE_SELECTION,
                       help='Compare different example selection methods')
    parser.add_argument('--pool-size', type=int, default=RANDOM_POOL_SIZE,
                       help='Size of random pool for random_then_sort selection method')
    
    args = parser.parse_args()
    
    # Check if corpus files exist
    if not os.path.exists(args.source) or not os.path.exists(args.target):
        logger.error(f"Corpus files not found. Please make sure {args.source} and {args.target} exist.")
        exit(1)
    
    # Create results directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created results directory: {args.output_dir}")
    
    # Initialize corpus
    corpus = PairCorpus(source_path=args.source, target_path=args.target)
    logger.info(f"Loaded corpus with {len(corpus.source_lines)} lines")
    
    # Create a quality estimator with multiple similarity measures
    estimator = QualityEstimator(
        similarity_measures=[
            TfidfCosineSimilarity(min_n=1, max_n=20)
        ],
        combination_mode="multiply"  # 'multiply' or 'average'
    )
    
    # List of error types (with proper typing)
    error_types: List[ErrorType] = ["replace", "insert", "delete", "swap"]  # type: ignore
    
    # Run sensitivity analysis
    if args.compare:
        logger.info(f"Running {args.level}-level error sensitivity analysis comparing example selection methods...")
        logger.info(f"Using random pool size of {args.pool_size} for random_then_sort method")
    else:
        logger.info(f"Running {args.level}-level error sensitivity analysis for 1-{args.max_errors} errors...")
        
    results = analyze_multiple_error_sensitivity(
        corpus=corpus,
        num_samples=args.samples,
        error_level=args.level,  # type: ignore
        error_types=error_types,
        max_errors=args.max_errors,
        estimator=estimator,
        example_size=args.example_size,
        num_workers=args.workers,
        compare_selection_methods=args.compare,
        random_pool_size=args.pool_size
    )
    
    # Print summary statistics
    if args.compare:
        # Print comparison results
        print(f"\n{args.level.capitalize()}-Level Error Analysis Results (Comparing Selection Methods):")
        
        search_results = results["search"]
        random_results = results["random"]
        random_then_sort_results = results["random_then_sort"]
        
        print("\nSearch-based Selection Method:")
        for error_count in range(1, args.max_errors + 1):
            ec_stats = search_results["by_error_count"][error_count]
            print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
                  f"average drop {ec_stats['average_quality_drop']:.4f}")
        
        print("\nRandom Selection Method:")
        for error_count in range(1, args.max_errors + 1):
            ec_stats = random_results["by_error_count"][error_count]
            print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
                  f"average drop {ec_stats['average_quality_drop']:.4f}")
        
        print(f"\nRandom-then-Sort Selection Method (pool size: {args.pool_size}):")
        for error_count in range(1, args.max_errors + 1):
            ec_stats = random_then_sort_results["by_error_count"][error_count]
            print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
                  f"average drop {ec_stats['average_quality_drop']:.4f}")
        
        # Calculate improvements
        search_single = search_results["by_error_count"][1]["downgrade_percent"]
        random_single = random_results["by_error_count"][1]["downgrade_percent"]
        random_then_sort_single = random_then_sort_results["by_error_count"][1]["downgrade_percent"]
        
        search_max = search_results["by_error_count"][args.max_errors]["downgrade_percent"]
        random_max = random_results["by_error_count"][args.max_errors]["downgrade_percent"]
        random_then_sort_max = random_then_sort_results["by_error_count"][args.max_errors]["downgrade_percent"]
        
        # Improvement of search vs random
        search_vs_random_single = ((search_single - random_single) / random_single) * 100 if random_single > 0 else 0
        search_vs_random_max = ((search_max - random_max) / random_max) * 100 if random_max > 0 else 0
        
        # Improvement of random_then_sort vs random
        random_then_sort_vs_random_single = ((random_then_sort_single - random_single) / random_single) * 100 if random_single > 0 else 0
        random_then_sort_vs_random_max = ((random_then_sort_max - random_max) / random_max) * 100 if random_max > 0 else 0
        
        print(f"\nImprovement Summary:")
        print(f"  Single error detection:")
        print(f"    - Search improves over Random by {search_vs_random_single:.1f}%")
        print(f"    - Random-then-Sort improves over Random by {random_then_sort_vs_random_single:.1f}%")
        print(f"  Multiple error detection ({args.max_errors} errors):")
        print(f"    - Search improves over Random by {search_vs_random_max:.1f}%")
        print(f"    - Random-then-Sort improves over Random by {random_then_sort_vs_random_max:.1f}%")
        
    else:
        # Print standard results
        print(f"\n{args.level.capitalize()}-Level Error Analysis Results:")
        print("\nDowngrade Percentage by Error Count:")
        for error_count in range(1, args.max_errors + 1):
            ec_stats = results["by_error_count"][error_count]
            print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
                  f"average drop {ec_stats['average_quality_drop']:.4f}")
        
        print("\nBy Error Type:")
        for error_type in error_types:
            print(f"  {error_type.capitalize()}")
            for i, error_count in enumerate(range(1, args.max_errors + 1)):
                et_stats = results["by_error_count"][error_count]["by_error_type"][error_type]
                if et_stats["count"] > 0:
                    downgrade_percent = (et_stats["downgrade_count"] / et_stats["count"]) * 100
                    avg_drop = 0.0
                    if et_stats["downgrade_count"] > 0:
                        avg_drop = et_stats["quality_drop"] / et_stats["downgrade_count"]
                    print(f"    {error_count} error{'s' if error_count > 1 else ''}: {downgrade_percent:.1f}% downgrade, "
                          f"average drop {avg_drop:.4f}")
    
    # Create and save visualizations
    logger.info("Creating visualizations...")
    figures = create_summary_visualization(results, args.level)
    
    # Save each figure
    for i, fig in enumerate(figures):
        if args.compare:
            if i == 0:
                # First figure is the comparison visualization
                output_file = os.path.join(args.output_dir, f"error_sensitivity_comparison.png")
            else:
                # Method-specific visualizations
                method = ["search", "random", "random_then_sort"][i-1]
                output_file = os.path.join(args.output_dir, f"error_sensitivity_{method}.png")
        else:
            # Standard visualization
            output_file = os.path.join(args.output_dir, f"error_sensitivity_results.png")
        
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    logger.info(f"{args.level.capitalize()}-level error analysis complete!") 