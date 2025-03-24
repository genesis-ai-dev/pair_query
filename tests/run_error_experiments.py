#!/usr/bin/env python3
"""
Script to run a series of experiments testing error sensitivity.

This script runs:
1. Baseline experiments
2. Sample size experiments
3. TF-IDF parameter experiments
4. Threshold optimization experiments
"""

import os
import sys
import logging
import argparse
import multiprocessing
from typing import List, Dict, Any, Tuple

# Add the parent directory to the Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the experiment framework
from error_experiments import (
    ExperimentConfig,
    run_baseline_experiment,
    run_sample_size_experiment,
    run_tfidf_param_experiment,
    find_optimal_thresholds
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOURCE_PATH = os.path.join(PARENT_DIR, "files/corpus/eng-engULB.txt")
TARGET_PATH = os.path.join(PARENT_DIR, "files/corpus/kos-kos.txt")
RESULTS_DIR = os.path.join(PARENT_DIR, "results/error_experiments")

def run_experiments(args):
    """Run the specified experiments."""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create a base experiment config
    base_config = ExperimentConfig(
        name="error_sensitivity_base",
        description="Error Sensitivity Analysis",
        corpus_source_path=args.source,
        corpus_target_path=args.target,
        error_level=args.error_level,
        max_errors=args.max_errors,
        example_size=args.examples,
        num_workers=args.workers,
        output_dir=RESULTS_DIR,
        tfidf_min_n=args.min_n,
        tfidf_max_n=args.max_n,
        combination_mode=args.combination_mode
    )
    
    # Run the selected experiments
    if args.baseline or args.all:
        logger.info("Running baseline experiment")
        baseline_results = run_baseline_experiment(
            base_config, 
            num_samples=args.samples
        )
    
    if args.sample_size or args.all:
        logger.info("Running sample size experiment")
        # Use smaller sizes for quick testing
        sample_sizes = [15, 25, 50] if args.quick else [15, 25, 50, 100, 500]
        
        # Create a specialized config for sample size tests
        sample_config = ExperimentConfig(
            name="sample_size_test",
            description="Sample Size Effect Analysis",
            corpus_source_path=args.source,
            corpus_target_path=args.target,
            error_level=args.error_level,
            max_errors=args.max_errors,
            example_size=args.examples,
            num_workers=args.workers,
            output_dir=RESULTS_DIR,
            tfidf_min_n=args.min_n,
            tfidf_max_n=args.max_n,
            combination_mode=args.combination_mode
        )
        
        sample_results = run_sample_size_experiment(
            sample_config, 
            sample_sizes=sample_sizes
        )
    
    if args.tfidf_params or args.all:
        logger.info("Running TF-IDF parameter experiment")
        # Use fewer combinations for quick testing
        if args.quick:
            min_n_values = [1, 2]
            max_n_values = [3, 10]
        else:
            min_n_values = [1, 2, 3]
            max_n_values = [1, 3, 5, 10, 20]
        
        # Create a specialized config for TF-IDF tests
        tfidf_config = ExperimentConfig(
            name="tfidf_param_test",
            description="TF-IDF Parameter Effect Analysis",
            corpus_source_path=args.source,
            corpus_target_path=args.target,
            error_level=args.error_level,
            max_errors=args.max_errors,
            example_size=args.examples,
            num_workers=args.workers,
            output_dir=RESULTS_DIR,
            tfidf_min_n=args.min_n,
            tfidf_max_n=args.max_n,
            combination_mode=args.combination_mode
        )
        
        tfidf_results = run_tfidf_param_experiment(
            tfidf_config,
            num_samples=args.samples,
            min_n_values=min_n_values,
            max_n_values=max_n_values
        )
    
    if args.thresholds or args.all:
        logger.info("Running threshold optimization experiment")
        
        # Create a specialized config for threshold tests
        threshold_config = ExperimentConfig(
            name="threshold_optimization",
            description="Quality Threshold Optimization",
            corpus_source_path=args.source,
            corpus_target_path=args.target,
            error_level=args.error_level,
            max_errors=args.max_errors,
            example_size=args.examples,
            num_workers=args.workers,
            output_dir=RESULTS_DIR,
            tfidf_min_n=args.min_n,
            tfidf_max_n=args.max_n,
            combination_mode=args.combination_mode
        )
        
        threshold_results = find_optimal_thresholds(
            threshold_config,
            num_samples=args.samples
        )
    
    logger.info("All experiments completed!")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run error sensitivity experiments")
    
    # Input corpus options
    parser.add_argument("--source", default=SOURCE_PATH, help="Source corpus file path")
    parser.add_argument("--target", default=TARGET_PATH, help="Target corpus file path")
    
    # Which experiments to run
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--baseline", action="store_true", help="Run baseline experiment")
    parser.add_argument("--sample-size", action="store_true", help="Run sample size experiment")
    parser.add_argument("--tfidf-params", action="store_true", help="Run TF-IDF parameter experiment")
    parser.add_argument("--thresholds", action="store_true", help="Run threshold optimization experiment")
    
    # Experiment parameters
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to test (default: 10)")
    parser.add_argument("--error-level", choices=["word", "character"], default="character", 
                       help="Level at which to introduce errors (default: character)")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum number of errors to introduce (default: 5)")
    parser.add_argument("--examples", type=int, default=20, help="Number of examples for quality estimation (default: 20)")
    parser.add_argument("--min-n", type=int, default=1, help="Default min_n for TF-IDF (default: 1)")
    parser.add_argument("--max-n", type=int, default=3, help="Default max_n for TF-IDF (default: 3)")
    parser.add_argument("--combination-mode", choices=["multiply", "average"], default="multiply",
                       help="How to combine multiple similarity measures (default: multiply)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                       help=f"Number of worker processes (default: {max(1, multiprocessing.cpu_count() - 1)})")
    parser.add_argument("--quick", action="store_true", help="Run with reduced parameters for quicker testing")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no specific experiment is selected, run all
    if not (args.all or args.baseline or args.sample_size or args.tfidf_params or args.thresholds):
        args.all = True
    
    # Check if corpus files exist
    if not os.path.exists(args.source) or not os.path.exists(args.target):
        logger.error(f"Corpus files not found. Please make sure {args.source} and {args.target} exist.")
        sys.exit(1)
    
    # Run the experiments
    run_experiments(args) 