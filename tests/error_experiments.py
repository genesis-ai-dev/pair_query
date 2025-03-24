#!/usr/bin/env python3
"""
Framework for running experiments on error sensitivity in quality estimation.

This module provides functions to:
1. Test how sample size affects error sensitivity
2. Test how TF-IDF parameters affect error sensitivity
3. Find optimal quality thresholds for error detection
"""

import os
import sys
import logging
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Literal, cast
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from tqdm.auto import tqdm

# Add the parent directory to the Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our modules
from error_sensitivity import (
    analyze_multiple_error_sensitivity,
    ErrorType,
    AnalysisResults
)
from pq.main import PairCorpus, QualityEstimator
from pq.similarity_measures import (
    NGramSimilarity,
    WordOverlapSimilarity,
    TfidfCosineSimilarity
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default path for saving results
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

@dataclass
class ExperimentConfig:
    """Configuration for an error sensitivity experiment."""
    name: str
    description: str
    corpus_source_path: str
    corpus_target_path: str
    error_level: Literal["word", "character"] = "character"
    error_types: List[ErrorType] = field(default_factory=lambda: ["replace", "insert", "delete", "swap"])
    max_errors: int = 5
    example_size: int = 20
    num_workers: int = max(1, os.cpu_count() or 4 - 1)  # Leave one CPU free
    output_dir: str = RESULTS_DIR
    tfidf_min_n: int = 1
    tfidf_max_n: int = 3
    combination_mode: str = "multiply"
    
    def get_output_path(self, suffix: str = "") -> str:
        """Get path for saving results."""
        filename = f"{self.name}{suffix}.png"
        return os.path.join(self.output_dir, filename)
    
    def get_json_path(self) -> str:
        """Get path for saving JSON results."""
        return os.path.join(self.output_dir, f"{self.name}_results.json")

def run_baseline_experiment(config: ExperimentConfig, num_samples: int = 10) -> AnalysisResults:
    """Run the baseline experiment with current settings.
    
    Args:
        config: Experiment configuration
        num_samples: Number of samples to test
        
    Returns:
        Analysis results
    """
    logger.info(f"Running baseline experiment '{config.name}' with {num_samples} samples")
    
    # Initialize corpus
    corpus = PairCorpus(source_path=config.corpus_source_path, target_path=config.corpus_target_path)
    
    # Create estimator with TfidfCosineSimilarity
    estimator = QualityEstimator(
        similarity_measures=[
            TfidfCosineSimilarity(min_n=config.tfidf_min_n, max_n=config.tfidf_max_n)
        ],
        combination_mode=config.combination_mode
    )
    
    # Run analysis
    results = analyze_multiple_error_sensitivity(
        corpus=corpus,
        num_samples=num_samples,
        error_level=config.error_level,  # type: ignore
        error_types=config.error_types,
        max_errors=config.max_errors,
        estimator=estimator,
        example_size=config.example_size,
        num_workers=config.num_workers
    )
    
    # Visualize and save results
    logger.info(f"Generating visualizations for '{config.name}'")
    visualize_results(results, config)
    
    return results

def run_sample_size_experiment(
    config: ExperimentConfig,
    sample_sizes: List[int] = [15, 25, 50, 100, 500]
) -> Dict[int, AnalysisResults]:
    """Run experiments with different sample sizes.
    
    Args:
        config: Experiment configuration
        sample_sizes: List of sample sizes to test
        
    Returns:
        Dictionary of results keyed by sample size
    """
    logger.info(f"Running sample size experiment with sizes: {sample_sizes}")
    
    results = {}
    for size in sample_sizes:
        logger.info(f"Testing with {size} samples")
        
        # Create a config copy with specific name for this size
        size_config = ExperimentConfig(
            name=f"{config.name}_samples_{size}",
            description=f"{config.description} (Sample Size: {size})",
            corpus_source_path=config.corpus_source_path,
            corpus_target_path=config.corpus_target_path,
            error_level=config.error_level,
            error_types=config.error_types,
            max_errors=config.max_errors,
            example_size=config.example_size,
            num_workers=config.num_workers,
            output_dir=config.output_dir,
            tfidf_min_n=config.tfidf_min_n,
            tfidf_max_n=config.tfidf_max_n,
            combination_mode=config.combination_mode
        )
        
        # Run experiment with this sample size
        # The visualize_results function will be called inside run_baseline_experiment
        results[size] = run_baseline_experiment(size_config, num_samples=size)
    
    return results

def run_tfidf_param_experiment(
    config: ExperimentConfig, 
    num_samples: int = 10,
    min_n_values: List[int] = [1, 2, 3],
    max_n_values: List[int] = [1, 3, 5, 10, 20]
) -> Dict[Tuple[int, int], AnalysisResults]:
    """Run experiments with different TF-IDF min_n and max_n parameters.
    
    Args:
        config: Experiment configuration
        num_samples: Number of samples to test
        min_n_values: List of min_n values to test
        max_n_values: List of max_n values to test
        
    Returns:
        Dictionary of results keyed by (min_n, max_n)
    """
    logger.info(f"Running TF-IDF parameter experiment with {len(min_n_values)}x{len(max_n_values)} combinations")
    
    results = {}
    for min_n in min_n_values:
        for max_n in max_n_values:
            # Skip invalid combinations
            if min_n > max_n:
                continue
                
            logger.info(f"Testing with min_n={min_n}, max_n={max_n}")
            
            # Create a config copy with specific name for these parameters
            param_config = ExperimentConfig(
                name=f"{config.name}_tfidf_min{min_n}_max{max_n}",
                description=f"{config.description} (TF-IDF: min_n={min_n}, max_n={max_n})",
                corpus_source_path=config.corpus_source_path,
                corpus_target_path=config.corpus_target_path,
                error_level=config.error_level,
                error_types=config.error_types,
                max_errors=config.max_errors,
                example_size=config.example_size,
                num_workers=config.num_workers,
                output_dir=config.output_dir,
                tfidf_min_n=min_n,
                tfidf_max_n=max_n,
                combination_mode=config.combination_mode
            )
            
            # Run experiment with these parameters
            # The visualize_results function will be called inside run_baseline_experiment
            results[(min_n, max_n)] = run_baseline_experiment(param_config, num_samples=num_samples)
    
    return results

def find_optimal_thresholds(
    config: ExperimentConfig,
    num_samples: int = 10
) -> Dict[int, Dict[str, Any]]:
    """Find optimal quality thresholds for detecting errors.
    
    Args:
        config: Experiment configuration
        num_samples: Number of samples to test
        
    Returns:
        Dictionary of results keyed by error count
    """
    logger.info(f"Finding optimal quality thresholds using {num_samples} samples")
    
    # Run the analysis to get all the test cases
    threshold_config = ExperimentConfig(
        name=f"{config.name}_thresholds",
        description=f"{config.description} (Threshold Analysis)",
        corpus_source_path=config.corpus_source_path,
        corpus_target_path=config.corpus_target_path,
        error_level=config.error_level,
        error_types=config.error_types,
        max_errors=config.max_errors,
        example_size=config.example_size,
        num_workers=config.num_workers,
        output_dir=config.output_dir,
        tfidf_min_n=config.tfidf_min_n,
        tfidf_max_n=config.tfidf_max_n,
        combination_mode=config.combination_mode
    )
    
    # Get the analysis results (visualization of these results is handled in run_baseline_experiment)
    results = run_baseline_experiment(threshold_config, num_samples=num_samples)
    
    # Extract quality scores and true labels for each error count
    threshold_results = {}
    
    for error_count in range(1, config.max_errors + 1):
        scores = []
        labels = []  # 1 = has error, 0 = no error
        
        # Go through all samples and collect quality scores and labels
        for sample in results["samples"]:
            original_quality = sample["original_quality"]
            
            # Skip samples that don't have results for this error count
            if error_count not in sample["error_results"]:
                continue
                
            for test in sample["error_results"][error_count]:
                # Add the degraded quality score
                scores.append(test["quality"])
                # Label is 1 because all test cases have errors
                labels.append(1)
            
            # Add the original quality score
            scores.append(original_quality)
            # Label is 0 because original has no error
            labels.append(0)
        
        # Convert to numpy arrays
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Calculate precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(labels, 1.0 - scores)
        average_precision = average_precision_score(labels, 1.0 - scores)
        
        # Calculate ROC curve
        fpr, tpr, thresholds_roc = roc_curve(labels, 1.0 - scores)
        roc_auc = auc(fpr, tpr)
        
        # Find the threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = 1.0 - (thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 0.5)
        
        # Calculate accuracy at the best threshold
        predicted = (scores <= best_threshold).astype(int)
        accuracy = np.mean(predicted == labels)
        
        # Store results
        threshold_results[error_count] = {
            "best_threshold": float(best_threshold),
            "best_f1": float(f1_scores[best_f1_idx]),
            "accuracy": float(accuracy),
            "precision": float(precision[best_f1_idx]),
            "recall": float(recall[best_f1_idx]),
            "average_precision": float(average_precision),
            "roc_auc": float(roc_auc),
            "num_samples": len(scores),
            "precision_curve": precision.tolist(),
            "recall_curve": recall.tolist(),
            "fpr_curve": fpr.tolist(),
            "tpr_curve": tpr.tolist()
        }
    
    # Create visualizations for the threshold analysis
    logger.info(f"Generating threshold visualizations for '{threshold_config.name}'")
    visualize_threshold_results(threshold_results, threshold_config)
    
    return threshold_results

def save_results(results: AnalysisResults, config: ExperimentConfig) -> None:
    """Save analysis results to a JSON file.
    
    Args:
        results: Analysis results to save
        config: Experiment configuration
    """
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare a simplified version of results for JSON
    json_results = {
        "experiment_name": config.name,
        "experiment_description": config.description,
        "error_level": config.error_level,
        "max_errors": config.max_errors,
        "tfidf_parameters": {
            "min_n": config.tfidf_min_n,
            "max_n": config.tfidf_max_n
        },
        "num_samples": len(results["samples"]),
        "timestamp": datetime.datetime.now().isoformat(),
        "error_counts": results["error_counts"],
        "by_error_count": {},
        "by_error_type": {}
    }
    
    # Convert error count stats
    for error_count in results["by_error_count"]:
        ec_stats = results["by_error_count"][error_count]
        json_results["by_error_count"][str(error_count)] = {
            "downgrade_percent": ec_stats["downgrade_percent"],
            "average_quality_drop": ec_stats["average_quality_drop"],
            "total_tests": ec_stats["total_tests"],
            "downgrade_count": ec_stats["downgrade_count"]
        }
    
    # Convert error type stats
    for error_type in results["by_error_type"]:
        json_results["by_error_type"][error_type] = {
            "downgrade_percent": results["by_error_type"][error_type]["downgrade_percent"],
            "avg_quality_drop": results["by_error_type"][error_type]["avg_quality_drop"]
        }
    
    # Save to JSON file
    with open(config.get_json_path(), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved results to {config.get_json_path()}")

def save_threshold_results(threshold_results: Dict[int, Dict[str, Any]], config: ExperimentConfig) -> None:
    """Save threshold analysis results to a JSON file.
    
    Args:
        threshold_results: Threshold analysis results
        config: Experiment configuration
    """
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare a simplified version of results for JSON
    json_results = {
        "experiment_name": config.name,
        "experiment_description": config.description,
        "error_level": config.error_level,
        "max_errors": config.max_errors,
        "tfidf_parameters": {
            "min_n": config.tfidf_min_n,
            "max_n": config.tfidf_max_n
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "thresholds": {}
    }
    
    # Include threshold results
    for error_count, result in threshold_results.items():
        # Make a copy without the curve data (which can be large)
        result_copy = result.copy()
        for key in ["precision_curve", "recall_curve", "fpr_curve", "tpr_curve"]:
            if key in result_copy:
                del result_copy[key]
        
        json_results["thresholds"][str(error_count)] = result_copy
    
    # Save to JSON file
    json_path = os.path.join(config.output_dir, f"{config.name}_thresholds.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved threshold results to {json_path}")

def visualize_results(results: AnalysisResults, config: ExperimentConfig) -> None:
    """Visualize analysis results and save figures.
    
    Args:
        results: Analysis results to visualize
        config: Experiment configuration
    """
    # Set up the style
    sns.set_style("whitegrid")
    
    # Create a plot with two subplots - top for percentage, bottom for quality drop
    fig = plt.figure(figsize=(14, 10))
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
    ax1.set_xlabel(f'Number of {config.error_level.capitalize()}-Level Errors', fontsize=13)
    ax1.set_ylabel('Quality Downgrade Percentage (%)', fontsize=13)
    ax1.set_title(f'Percentage of Translations with Quality Downgrade\nby Number of {config.error_level.capitalize()}-Level Errors', 
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
    ax2.set_xlabel(f'Number of {config.error_level.capitalize()}-Level Errors', fontsize=13)
    ax2.set_ylabel('Average Quality Drop', fontsize=13)
    ax2.set_title(f'Average Quality Score Drop\nby Number of {config.error_level.capitalize()}-Level Errors', 
                fontsize=16, fontweight='bold')
    ax2.set_xticks(error_counts)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add a summary text box
    summary_text = (
        f"{config.description}\n"
        f"• Samples tested: {len(results['samples'])}\n"
        f"• TF-IDF parameters: min_n={config.tfidf_min_n}, max_n={config.tfidf_max_n}\n"
        f"• Single error detection rate: {overall_percents[0]:.1f}%\n"
        f"• Average quality drop (5 errors): {overall_drops[-1]:.4f}\n"
        f"• Most sensitive to: {error_types[np.argmax([results['by_error_type'][et]['avg_quality_drop'][-1] for et in error_types])].capitalize()} errors"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=13, 
               bbox={"facecolor":"lightyellow", "alpha":0.9, "pad":10, "edgecolor":"orange"})
    
    # Add overall title and adjust layout
    plt.suptitle(f"{config.description}", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Get the output path
    output_path = config.get_output_path()
    logger.info(f"Will save figure to: {output_path}")
    
    # Save the figure
    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save figure: {str(e)}")
    
    # Also save results to JSON
    save_results(results, config)
    
    # Close the figure to free memory
    plt.close(fig)

def visualize_threshold_results(threshold_results: Dict[int, Dict[str, Any]], config: ExperimentConfig) -> None:
    """Visualize threshold analysis results and save figures.
    
    Args:
        threshold_results: Threshold analysis results
        config: Experiment configuration
    """
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11  # Increase default font size
    
    # Create a figure for threshold values
    fig1 = plt.figure(figsize=(12, 9))
    
    # Extract data
    error_counts = sorted(threshold_results.keys())
    thresholds = [threshold_results[ec]["best_threshold"] for ec in error_counts]
    f1_scores = [threshold_results[ec]["best_f1"] for ec in error_counts]
    accuracies = [threshold_results[ec]["accuracy"] for ec in error_counts]
    
    # Create the main plot with two subplots side by side
    gs = plt.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)
    
    # First subplot - Threshold values
    ax1 = plt.subplot(gs[0])
    
    # Plot thresholds
    threshold_line = ax1.plot(error_counts, thresholds, 'b-', linewidth=3, markersize=12, 
                             marker='o', label="Quality Threshold")
    
    # Annotate thresholds with clear labels
    for i, threshold in enumerate(thresholds):
        ax1.annotate(f"{threshold:.4f}", 
                   xy=(error_counts[i], threshold + 0.003),
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.6))
    
    # Create a secondary y-axis for accuracy
    ax2 = ax1.twinx()
    accuracy_line = ax2.plot(error_counts, accuracies, 'g-', linewidth=3, markersize=12,
                            marker='s', label="Accuracy")
    
    # Annotate accuracy values
    for i, accuracy in enumerate(accuracies):
        ax2.annotate(f"{accuracy:.0%}", 
                   xy=(error_counts[i], accuracy + 0.02),
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.6))
    
    # Set labels and title with clear language
    ax1.set_xlabel('Number of Errors', fontsize=14)
    ax1.set_ylabel('Quality Threshold', fontsize=14, color='blue')
    ax2.set_ylabel('Accuracy', fontsize=14, color='green')
    ax1.set_title(f'Quality Thresholds for {config.error_level.capitalize()}-Level Errors', 
                fontsize=16, fontweight='bold')
    
    # Set x-ticks and limits
    ax1.set_xticks(error_counts)
    ax1.set_xlim(min(error_counts) - 0.5, max(error_counts) + 0.5)
    
    # Customize threshold axis
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Customize accuracy axis
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Combine legends from both axes
    lines = threshold_line + accuracy_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=13, framealpha=0.9)
    
    # Second subplot - What this means in practice (a simple interpretation guide)
    ax3 = plt.subplot(gs[1])
    ax3.axis('off')  # Turn off axis
    
    # Create explanatory text for non-technical users
    explanation_text = [
        "What does this mean?",
        "",
        f"• The blue line shows the quality score",
        f"  threshold that best separates good",
        f"  translations from those with errors.",
        "",
        f"• Translations with quality scores",
        f"  BELOW the threshold should be",
        f"  flagged for review.",
        "",
        f"• The green line shows the percentage",
        f"  of translations correctly classified",
        f"  at each threshold.",
        "",
        f"For detecting ONE error:",
        f"• Use threshold: {thresholds[0]:.4f}",
        f"• Expected accuracy: {accuracies[0]:.0%}",
        "",
        f"For detecting MULTIPLE errors:",
        f"• Use threshold: {thresholds[-1]:.4f}",
        f"• Expected accuracy: {accuracies[-1]:.0%}"
    ]
    
    y_pos = 0.95
    ax3.text(0.05, y_pos, '\n'.join(explanation_text), 
            va='top', ha='left', fontsize=13,
            bbox=dict(boxstyle="round,pad=1.0", 
                     fc='lightyellow', ec="orange", alpha=0.9))
    
    # Add overall title
    plt.suptitle(f"Finding the Right Quality Threshold\nfor Detecting {config.error_level.capitalize()}-Level Translation Errors", 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save the threshold figure
    threshold_path = config.get_output_path("_thresholds")
    logger.info(f"Will save threshold visualization to: {threshold_path}")
    try:
        fig1.savefig(threshold_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold visualization to {threshold_path}")
    except Exception as e:
        logger.error(f"Failed to save threshold figure: {str(e)}")
    
    # Save the threshold results to JSON
    save_threshold_results(threshold_results, config)
    
    # Create a single summary visualization for precision-recall instead of per-error graphs
    fig2 = plt.figure(figsize=(12, 7))
    
    # Set up colors for different error counts
    colors = plt.cm.viridis(np.linspace(0, 1, len(error_counts)))
    
    # Line styles for different metrics
    line_styles = ['-', '--']
    
    # Plot PR and ROC curves for each error count on a single plot
    for i, error_count in enumerate(error_counts):
        result = threshold_results[error_count]
        color = colors[i]
        
        # Label for all curves from this error count
        label_base = f"{error_count} Error{'s' if error_count > 1 else ''}"
        
        # Plot precision-recall curve
        plt.plot(result["recall_curve"], result["precision_curve"], 
                 line_styles[0], color=color, linewidth=2.5,
                 label=f"{label_base} (F1={result['best_f1']:.2f})")
        
    # Add a diagonal line representing random performance
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('True Positive Rate (Recall)', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision vs. Recall for Different Error Counts', fontsize=16, fontweight='bold')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=12, framealpha=0.9)
    
    # Add explanatory text
    explanation_text = (
        "This chart shows how well the system detects errors:\n"
        "• Higher curves are better\n"
        "• X-axis: % of actual errors detected\n"
        "• Y-axis: % of flagged translations that actually have errors"
    )
    
    plt.figtext(0.5, 0.01, explanation_text, ha="center", fontsize=13, 
               bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9, ec="orange"))
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    
    # Save the curves figure
    curves_path = config.get_output_path("_precision_recall")
    logger.info(f"Will save precision-recall visualization to: {curves_path}")
    try:
        fig2.savefig(curves_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall visualization to {curves_path}")
    except Exception as e:
        logger.error(f"Failed to save precision-recall figure: {str(e)}")
    
    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)

if __name__ == "__main__":
    # This script should be imported, not run directly
    print("This is a module for running error sensitivity experiments. Import it in another script.") 