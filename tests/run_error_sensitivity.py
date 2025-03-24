#!/usr/bin/env python3
"""
Script to analyze how the Quality Estimator responds to multiple errors.
This script tests error counts from 1 to 5 for each error type and creates a simple visualization.
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from error_sensitivity import analyze_multiple_error_sensitivity
from extract import PairCorpus, QualityEstimator
from similarity_measures import (
    NGramSimilarity,
    WordOverlapSimilarity,
    TfidfCosineSimilarity
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values
SOURCE_PATH = "corpus/eng-engULB.txt"
TARGET_PATH = "corpus/kos-kos.txt"
ERROR_LEVEL = "character"  # Options: "word" or "character"
NUM_SAMPLES = 30      # Number of samples to test
MAX_ERRORS = 5        # Maximum number of errors to introduce
EXAMPLE_SIZE = 20     # Number of examples for quality estimation
OUTPUT_FILE = "results/error_sensitivity_results.png"  # Output file for visualization

def create_summary_visualization(results, error_level):
    """Create a clear, informative visualization of the error sensitivity results."""
    # Set up the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Create a plot with two subplots - top for percentage, bottom for quality drop
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Extract common data
    error_counts = results["error_counts"]
    error_types = list(results["by_error_type"].keys())
    colors = sns.color_palette("husl", len(error_types))
    
    # Top subplot: Downgrade percentage by error count
    ax1 = plt.subplot(gs[0])
    
    # Plot lines for each error type with custom colors
    for i, error_type in enumerate(error_types):
        downgrade_percents = results["by_error_type"][error_type]["downgrade_percent"]
        ax1.plot(error_counts, downgrade_percents, 'o-', 
                linewidth=2.5, markersize=10, 
                color=colors[i], label=f"{error_type.capitalize()}")
    
    # Calculate and plot overall average with black
    overall_percents = [results["by_error_count"][ec]["downgrade_percent"] for ec in error_counts]
    ax1.plot(error_counts, overall_percents, 'ko-', 
            linewidth=3, markersize=12, label="Overall Average")
    
    # Add value labels above the points
    for i, percent in enumerate(overall_percents):
        ax1.annotate(f"{percent:.1f}%", 
                   xy=(error_counts[i], percent + 2),
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    # Set limits, labels and legend
    ax1.set_ylim(0, 105)
    ax1.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=13)
    ax1.set_ylabel('Quality Downgrade Percentage (%)', fontsize=13)
    ax1.set_title(f'Percentage of Translations with Quality Downgrade\nby Number of {error_level.capitalize()}-Level Errors', 
                fontsize=16, fontweight='bold')
    ax1.set_xticks(error_counts)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Bottom subplot: Average quality drop by error count
    ax2 = plt.subplot(gs[1])
    
    # Extract quality drops for each error type with the same colors as above
    for i, error_type in enumerate(error_types):
        quality_drops = results["by_error_type"][error_type]["avg_quality_drop"]
        ax2.plot(error_counts, quality_drops, 'o-',
                linewidth=2.5, markersize=10,
                color=colors[i], label=f"{error_type.capitalize()}")
    
    # Calculate and plot overall average with black
    overall_drops = [results["by_error_count"][ec]["average_quality_drop"] for ec in error_counts]
    ax2.plot(error_counts, overall_drops, 'ko-', 
            linewidth=3, markersize=12, label="Overall Average")
    
    # Add value labels above the points
    for i, drop in enumerate(overall_drops):
        ax2.annotate(f"{drop:.4f}", 
                   xy=(error_counts[i], drop + max(overall_drops) * 0.05),
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    # Set labels and legend
    ax2.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=13)
    ax2.set_ylabel('Average Quality Drop', fontsize=13)
    ax2.set_title(f'Average Quality Score Drop\nby Number of {error_level.capitalize()}-Level Errors', 
                fontsize=16, fontweight='bold')
    ax2.set_xticks(error_counts)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add a summary text box
    summary_text = (
        f"{error_level.capitalize()}-Level Error Analysis Summary:\n"
        f"• Samples tested: {len(results['samples'])}\n"
        f"• Single error detection rate: {overall_percents[0]:.1f}%\n"
        f"• Average quality drop (5 errors): {overall_drops[-1]:.4f}\n"
        f"• Most sensitive to: {error_types[np.argmax([results['by_error_type'][et]['avg_quality_drop'][-1] for et in error_types])].capitalize()} errors"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=13, 
               bbox={"facecolor":"lightyellow", "alpha":0.9, "pad":10, "edgecolor":"orange"})
    
    # Add overall title and adjust layout
    plt.suptitle(f"Quality Estimation Sensitivity to {error_level.capitalize()}-Level Errors", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return plt.gcf()

if __name__ == "__main__":
    # Check if corpus files exist
    if not os.path.exists(SOURCE_PATH) or not os.path.exists(TARGET_PATH):
        logger.error(f"Corpus files not found. Please make sure {SOURCE_PATH} and {TARGET_PATH} exist.")
        exit(1)
    
    # Initialize corpus
    corpus = PairCorpus(source_path=SOURCE_PATH, target_path=TARGET_PATH)
    logger.info(f"Loaded corpus with {len(corpus.source_lines)} lines")
    
    # Create a quality estimator with multiple similarity measures
    estimator = QualityEstimator(
        similarity_measures=[
            NGramSimilarity(max_n=3),
            WordOverlapSimilarity(), 
            TfidfCosineSimilarity(min_n=1, max_n=3)
        ],
        combination_mode="multiply"  # 'multiply' or 'average'
    )
    
    # Run sensitivity analysis
    logger.info(f"Running {ERROR_LEVEL}-level error sensitivity analysis for 1-{MAX_ERRORS} errors...")
    results = analyze_multiple_error_sensitivity(
        corpus=corpus,
        num_samples=NUM_SAMPLES,
        error_level=ERROR_LEVEL,
        error_types=["replace", "insert", "delete", "swap"],
        max_errors=MAX_ERRORS,
        estimator=estimator,
        example_size=EXAMPLE_SIZE
    )
    
    # Print summary statistics
    print(f"\n{ERROR_LEVEL.capitalize()}-Level Error Analysis Results:")
    print("\nDowngrade Percentage by Error Count:")
    for error_count in range(1, MAX_ERRORS + 1):
        ec_stats = results["by_error_count"][error_count]
        print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
              f"average drop {ec_stats['average_quality_drop']:.4f}")
    
    print("\nBy Error Type:")
    for error_type in ["replace", "insert", "delete", "swap"]:
        print(f"  {error_type.capitalize()}")
        for i, error_count in enumerate(range(1, MAX_ERRORS + 1)):
            et_stats = results["by_error_count"][error_count]["by_error_type"][error_type]
            if et_stats["count"] > 0:
                downgrade_percent = (et_stats["downgrade_count"] / et_stats["count"]) * 100
                avg_drop = 0.0
                if et_stats["downgrade_count"] > 0:
                    avg_drop = et_stats["quality_drop"] / et_stats["downgrade_count"]
                print(f"    {error_count} error{'s' if error_count > 1 else ''}: {downgrade_percent:.1f}% downgrade, "
                      f"average drop {avg_drop:.4f}")
    
    # Create and save visualization
    logger.info("Creating visualization...")
    fig = create_summary_visualization(results, ERROR_LEVEL)
    
    # Save the figure
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {OUTPUT_FILE}")
    
    logger.info(f"{ERROR_LEVEL.capitalize()}-level error analysis complete!") 