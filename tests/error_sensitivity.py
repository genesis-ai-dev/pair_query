import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, TypedDict, cast
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import argparse
import logging
import seaborn as sns
import os
import sys
from functools import partial

# Add the parent directory to the Python path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required classes from extract.py
from pq.main import PairCorpus, QualityEstimator
from pq.similarity_measures import (
    SimilarityMeasure,
    NGramSimilarity,
    WordOverlapSimilarity, 
    LongestSubstringSimilarity,
    TfidfCosineSimilarity
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions to improve type checking
ErrorType = Literal["replace", "insert", "delete", "swap"]

class ErrorTypeStats(TypedDict):
    count: int
    downgrade_count: int
    quality_drop: float

class ErrorCountStats(TypedDict):
    downgrade_count: int
    total_tests: int
    downgrade_percent: float
    total_quality_drop: float
    average_quality_drop: float
    by_error_type: Dict[str, ErrorTypeStats]

class ErrorTestResult(TypedDict):
    error_type: str
    modified_text: str
    quality: float
    quality_drop: float
    is_downgrade: bool

class SampleResults(TypedDict):
    source: str
    reference: str
    original_quality: float
    error_results: Dict[int, List[ErrorTestResult]]

class AnalysisResults(TypedDict):
    error_counts: List[int]
    samples: List[SampleResults]
    by_error_count: Dict[int, ErrorCountStats]
    by_error_type: Dict[str, Dict[str, List[float]]]

class SingleErrorGenerator:
    """Class for introducing errors into text."""
    
    @staticmethod
    def introduce_word_errors(text: str, error_type: Optional[ErrorType] = None, 
                             num_errors: int = 1) -> str:
        """Introduce multiple word-level errors.
        
        Args:
            text: The original text
            error_type: The type of error to introduce (replace, insert, delete, swap)
                        If None, a random error type will be chosen.
            num_errors: Number of errors to introduce
        """
        words = text.split()
        if not words or num_errors <= 0:
            return text
            
        # If no error type specified, choose randomly
        if error_type is None:
            error_type = random.choice(["replace", "insert", "delete", "swap"])
            
        # Make a copy of the words list to modify
        modified_words = words.copy()
        
        # Introduce multiple errors
        for _ in range(num_errors):
            # For very short texts or if we've deleted too many words, stop adding errors
            if len(modified_words) < 2:
                break
                
            if error_type == "replace":
                # Replace a random word with a random string
                idx = random.randint(0, len(modified_words) - 1)
                word_len = len(modified_words[idx])
                modified_words[idx] = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') 
                                             for _ in range(max(3, word_len)))
                    
            elif error_type == "insert":
                # Insert a random word
                idx = random.randint(0, len(modified_words))
                random_word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') 
                                    for _ in range(random.randint(3, 8)))
                modified_words.insert(idx, random_word)
                
            elif error_type == "delete" and len(modified_words) > 1:
                # Delete a random word
                idx = random.randint(0, len(modified_words) - 1)
                modified_words.pop(idx)
                
            elif error_type == "swap" and len(modified_words) > 1:
                # Swap two adjacent words
                if len(modified_words) == 2:
                    idx = 0
                else:
                    idx = random.randint(0, len(modified_words) - 2)
                modified_words[idx], modified_words[idx + 1] = modified_words[idx + 1], modified_words[idx]
        
        return ' '.join(modified_words)
    
    @staticmethod
    def introduce_character_errors(text: str, error_type: Optional[ErrorType] = None,
                                 num_errors: int = 1) -> str:
        """Introduce multiple character-level errors.
        
        Args:
            text: The original text
            error_type: The type of error to introduce (replace, insert, delete, swap)
                        If None, a random error type will be chosen.
            num_errors: Number of errors to introduce
        """
        if not text or num_errors <= 0:
            return text
            
        # If no error type specified, choose randomly
        if error_type is None:
            error_type = random.choice(["replace", "insert", "delete", "swap"])
            
        # Make a copy of the text to modify
        chars = list(text)
        
        # Introduce multiple errors
        for _ in range(num_errors):
            # Stop if the text has become too short
            if len(chars) < 2:
                break
                
            if error_type == "replace" and chars:
                # Replace a random character
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz ')
                    
            elif error_type == "insert" and chars:
                # Insert a random character
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz '))
                
            elif error_type == "delete" and len(chars) > 1:
                # Delete a random character
                idx = random.randint(0, len(chars) - 1)
                chars.pop(idx)
                
            elif error_type == "swap" and len(chars) > 1:
                # Swap two adjacent characters
                if len(chars) == 2:
                    idx = 0
                else:
                    idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        
        return ''.join(chars)

def analyze_single_sample(sample_info: Tuple[str, str, int, QualityEstimator, PairCorpus, Literal["word", "character"], 
                                           List[ErrorType], int, int, str, int]) -> SampleResults:
    """Analyze a single sample with multiple error types and counts.
    
    Args:
        sample_info: Tuple containing (source_text, reference_text, idx, estimator, corpus, 
                                       error_level, error_types, max_errors, example_size,
                                       example_selection, random_pool_size)
    
    Returns:
        Analysis results for this sample
    """
    source, reference, idx, estimator, corpus, error_level, error_types, max_errors, example_size, example_selection, random_pool_size = sample_info
    
    # Evaluate original quality once
    original_quality = estimator.evaluate_translation(
        source, reference, corpus, 
        sample_size=example_size,
        example_selection=example_selection,
        random_pool_size=random_pool_size
    )
    
    sample_results: SampleResults = {
        "source": source,
        "reference": reference,
        "original_quality": original_quality,
        "error_results": {}
    }
    
    # For each error count
    for error_count in range(1, max_errors + 1):
        error_count_results: List[ErrorTestResult] = []
        
        # For each error type
        for error_type in error_types:
            # Introduce errors based on level
            typed_error_type = cast(ErrorType, error_type)  # Cast to help type checking
            
            if error_level == "word":
                modified = SingleErrorGenerator.introduce_word_errors(
                    reference, typed_error_type, num_errors=error_count
                )
            else:  # character level
                modified = SingleErrorGenerator.introduce_character_errors(
                    reference, typed_error_type, num_errors=error_count
                )
            
            # Skip if no change was made
            if modified == reference:
                continue
                
            # Evaluate modified quality
            modified_quality = estimator.evaluate_translation(
                source, modified, corpus, 
                sample_size=example_size,
                example_selection=example_selection,
                random_pool_size=random_pool_size
            )
            
            quality_drop = original_quality - modified_quality
            is_downgrade = quality_drop > 0
            
            # Record result for this test
            error_count_results.append({
                "error_type": error_type,
                "modified_text": modified,
                "quality": modified_quality,
                "quality_drop": quality_drop,
                "is_downgrade": is_downgrade
            })
        
        # Store results for this error count
        sample_results["error_results"][error_count] = error_count_results
    
    return sample_results

def analyze_multiple_error_sensitivity(
    corpus: PairCorpus,
    num_samples: int = 20,
    error_level: Literal["word", "character"] = "word",
    error_types: Optional[List[ErrorType]] = None,
    max_errors: int = 5,
    estimator: Optional[QualityEstimator] = None,
    example_size: int = 10,
    num_workers: int = 4,
    compare_selection_methods: bool = False,
    random_pool_size: int = 1000
) -> Union[AnalysisResults, Dict[str, AnalysisResults]]:
    """
    Analyze sensitivity of quality estimation to multiple errors.
    
    Args:
        corpus: The parallel corpus to use
        num_samples: Number of samples to test
        error_level: Level at which to introduce errors ("word" or "character")
        error_types: List of error types to test ("replace", "insert", "delete", "swap")
                    If None, all error types will be tested.
        max_errors: Maximum number of errors to introduce
        estimator: Quality estimator to use
        example_size: Number of examples to use for quality estimation
        num_workers: Number of worker processes for parallelization
        compare_selection_methods: Whether to compare random vs search example selection
        random_pool_size: For 'random_then_sort', size of initial random pool before filtering
    
    Returns:
        If compare_selection_methods is False: Dictionary containing analysis results
        If compare_selection_methods is True: Dictionary with 'search' and 'random' results
    """
    # Use default error types if none provided
    if error_types is None:
        error_types = ["replace", "insert", "delete", "swap"]
    
    # Create default estimator if none provided
    if estimator is None:
        estimator = QualityEstimator(
            similarity_measures=[
                NGramSimilarity(),
                WordOverlapSimilarity(),
                TfidfCosineSimilarity()
            ]
        )
    
    # Find valid test samples (non-empty sentences)
    valid_indices = [i for i in range(len(corpus.source_lines)) 
                   if len(corpus.source_lines[i].strip()) > 10 
                   and len(corpus.target_lines[i].strip()) > 10]
    
    if len(valid_indices) < num_samples:
        logger.warning(f"Not enough valid samples. Need {num_samples}, found {len(valid_indices)}")
        num_samples = len(valid_indices)
    
    test_indices = random.sample(valid_indices, num_samples)
    
    # Define selection methods to test
    selection_methods = ["search", "random", "random_then_sort"] if compare_selection_methods else ["search"]
    results_by_method: Dict[str, AnalysisResults] = {}
    
    for selection_method in selection_methods:
        logger.info(f"Starting analysis with {selection_method} example selection method")
        
        # Initialize results structure
        results: AnalysisResults = {
            "error_counts": list(range(1, max_errors + 1)),
            "samples": [],
            "by_error_count": {},
            "by_error_type": {et: {"downgrade_percent": [], "avg_quality_drop": []} for et in error_types}
        }
        
        # Initialize per-error-count results
        for error_count in range(1, max_errors + 1):
            results["by_error_count"][error_count] = {
                "downgrade_count": 0,
                "total_tests": 0,
                "downgrade_percent": 0.0,
                "total_quality_drop": 0.0,
                "average_quality_drop": 0.0,
                "by_error_type": {et: {"count": 0, "downgrade_count": 0, "quality_drop": 0.0} for et in error_types}
            }
            
        # Create executor with multiple processes
        test_samples = []
        for i, idx in enumerate(test_indices):
            source = corpus.source_lines[idx].strip()
            reference = corpus.target_lines[idx].strip()
            
            # Create a sample info tuple - include selection_method and random_pool_size
            sample_info = (source, reference, idx, estimator, corpus, error_level, 
                          error_types, max_errors, example_size, selection_method, random_pool_size)
            test_samples.append(sample_info)
            
        # Run analysis in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Update the analyze_single_sample function call to pass selection_method
            futures = [executor.submit(analyze_single_sample, sample_info) for sample_info in test_samples]
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Analyzing samples ({selection_method})", unit="sample"):
                sample_result = future.result()
                results["samples"].append(sample_result)
                
                # Aggregate results by error count and type
                for error_count in sample_result["error_results"]:
                    ec_result = results["by_error_count"][error_count]
                    
                    for test in sample_result["error_results"][error_count]:
                        ec_result["total_tests"] += 1
                        error_type = test["error_type"]
                        
                        # Update error type stats
                        et_stats = ec_result["by_error_type"][error_type]
                        et_stats["count"] += 1
                        
                        # If quality is lower (error detected)
                        if test["is_downgrade"]:
                            ec_result["downgrade_count"] += 1
                            ec_result["total_quality_drop"] += test["quality_drop"]
                            
                            et_stats["downgrade_count"] += 1
                            et_stats["quality_drop"] += test["quality_drop"]
                            
        # Calculate final statistics
        for error_count in range(1, max_errors + 1):
            ec_result = results["by_error_count"][error_count]
            
            # Calculate percentage and average drop
            if ec_result["total_tests"] > 0:
                ec_result["downgrade_percent"] = (ec_result["downgrade_count"] / ec_result["total_tests"]) * 100
                
            if ec_result["downgrade_count"] > 0:
                ec_result["average_quality_drop"] = ec_result["total_quality_drop"] / ec_result["downgrade_count"]
                
        # Calculate per-error-type statistics
        for error_type in error_types:
            downgrade_percent_by_count = []
            avg_drop_by_count = []
            
            for error_count in range(1, max_errors + 1):
                et_stats = results["by_error_count"][error_count]["by_error_type"][error_type]
                
                # Calculate percentage for this error count and type
                if et_stats["count"] > 0:
                    downgrade_percent = (et_stats["downgrade_count"] / et_stats["count"]) * 100
                else:
                    downgrade_percent = 0.0
                    
                # Calculate average drop for this error count and type
                if et_stats["downgrade_count"] > 0:
                    avg_drop = et_stats["quality_drop"] / et_stats["downgrade_count"]
                else:
                    avg_drop = 0.0
                    
                downgrade_percent_by_count.append(downgrade_percent)
                avg_drop_by_count.append(avg_drop)
                
            # Store the lists for this error type
            results["by_error_type"][error_type]["downgrade_percent"] = downgrade_percent_by_count
            results["by_error_type"][error_type]["avg_quality_drop"] = avg_drop_by_count
            
        # Store the results for this selection method
        results_by_method[selection_method] = results
    
    # Return the appropriate result structure
    if compare_selection_methods:
        return results_by_method
    else:
        return results_by_method["search"]

def visualize_multiple_error_results(
    results: Union[AnalysisResults, Dict[str, AnalysisResults]], 
    error_level: str = "word"
) -> List[plt.Figure]:
    """Create visualizations of error sensitivity analysis results.
    
    Args:
        results: Error sensitivity analysis results, either a single result 
                or a dictionary with results for different selection methods
        error_level: Level at which errors were introduced
        
    Returns:
        List of figure objects
    """
    # Set up the style
    sns.set_style("whitegrid")
    figures = []
    
    # Check if results contains comparison data
    if isinstance(results, dict) and "search" in results and "random" in results:
        # Create a comparison visualization
        fig_comparison = _create_comparison_visualization(results, error_level)
        figures.append(fig_comparison)
        
        # Create individual visualizations for each method
        for method, method_results in results.items():
            fig = _create_error_sensitivity_visualization(method_results, error_level, method)
            figures.append(fig)
    else:
        # Single results case
        single_results = cast(AnalysisResults, results)
        fig = _create_error_sensitivity_visualization(single_results, error_level)
        figures.append(fig)
    
    return figures

def _create_error_sensitivity_visualization(
    results: AnalysisResults, 
    error_level: str, 
    selection_method: str = "search"
) -> plt.Figure:
    """Create visualization for error sensitivity results with a single selection method.
    
    Args:
        results: Error sensitivity analysis results
        error_level: Level at which errors were introduced
        selection_method: The example selection method used
        
    Returns:
        Figure object
    """
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
    ax2.set_title(f'Average Quality Drop\nby Number of {error_level.capitalize()}-Level Errors', 
                fontsize=16, fontweight='bold')
    ax2.set_xticks(error_counts)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add a summary text box
    summary_text = (
        f"{error_level.capitalize()}-Level Error Analysis Summary (using {selection_method.capitalize()} examples):\n"
        f"• Samples tested: {len(results['samples'])}\n"
        f"• Single error detection rate: {overall_percents[0]:.1f}%\n"
        f"• Average quality drop (5 errors): {overall_drops[-1]:.4f}\n"
        f"• Most sensitive to: {error_types[np.argmax([results['by_error_type'][et]['avg_quality_drop'][-1] for et in error_types])].capitalize()} errors"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=13, 
               bbox={"facecolor":"lightyellow", "alpha":0.9, "pad":10, "edgecolor":"orange"})
    
    # Add overall title and adjust layout
    plt.suptitle(f"Quality Estimation Sensitivity to {error_level.capitalize()}-Level Errors\n(using {selection_method.capitalize()} Examples)", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    
    return fig

def _create_comparison_visualization(
    results: Dict[str, AnalysisResults], 
    error_level: str
) -> plt.Figure:
    """Create visualization comparing different example selection methods.
    
    Args:
        results: Dictionary with results for different selection methods
        error_level: Level at which errors were introduced
        
    Returns:
        Figure object
    """
    # Create a figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Extract common data
    error_counts = results["search"]["error_counts"]
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)  # Downgrade percentage comparison
    ax2 = plt.subplot(1, 2, 2)  # Average quality drop comparison
    
    # Set up styles for the methods
    styles = {
        "search": {"color": "blue", "marker": "o", "linestyle": "-", "label": "Search-based Examples"},
        "random": {"color": "red", "marker": "s", "linestyle": "--", "label": "Random Examples"},
        "random_then_sort": {"color": "green", "marker": "^", "linestyle": "-.", "label": "Random-then-Sort Examples"}
    }
    
    # Plot downgrade percentage comparison
    for method, style in styles.items():
        if method not in results:
            continue
            
        method_results = results[method]
        overall_percents = [method_results["by_error_count"][ec]["downgrade_percent"] for ec in error_counts]
        
        line = ax1.plot(error_counts, overall_percents, 
                      marker=style["marker"], color=style["color"], 
                      linestyle=style["linestyle"], linewidth=3, markersize=10,
                      label=style["label"])
        
        # Add value labels
        for i, percent in enumerate(overall_percents):
            ax1.annotate(f"{percent:.1f}%", 
                       xy=(error_counts[i], percent + 1),
                       ha='center', va='bottom',
                       fontsize=10, color=style["color"])
    
    # Configure downgrade percentage subplot
    ax1.set_ylim(0, 105)
    ax1.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=13)
    ax1.set_ylabel('Quality Downgrade Percentage (%)', fontsize=13)
    ax1.set_title(f'Downgrade Percentage Comparison\nBy Example Selection Method', 
                fontsize=15, fontweight='bold')
    ax1.set_xticks(error_counts)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Plot average quality drop comparison
    for method, style in styles.items():
        if method not in results:
            continue
            
        method_results = results[method]
        overall_drops = [method_results["by_error_count"][ec]["average_quality_drop"] for ec in error_counts]
        
        line = ax2.plot(error_counts, overall_drops, 
                      marker=style["marker"], color=style["color"], 
                      linestyle=style["linestyle"], linewidth=3, markersize=10,
                      label=style["label"])
        
        # Add value labels
        for i, drop in enumerate(overall_drops):
            ax2.annotate(f"{drop:.4f}", 
                       xy=(error_counts[i], drop + 0.01),
                       ha='center', va='bottom',
                       fontsize=10, color=style["color"])
    
    # Configure quality drop subplot
    ax2.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=13)
    ax2.set_ylabel('Average Quality Drop', fontsize=13)
    ax2.set_title(f'Quality Drop Comparison\nBy Example Selection Method', 
                fontsize=15, fontweight='bold')
    ax2.set_xticks(error_counts)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Calculate improvement percentages for the summary
    search_results = results["search"]
    random_results = results["random"]
    random_then_sort_results = results.get("random_then_sort", None)
    
    # Basic comparison between search and random
    search_single_error = search_results["by_error_count"][1]["downgrade_percent"]
    random_single_error = random_results["by_error_count"][1]["downgrade_percent"]
    
    search_max_error = search_results["by_error_count"][max(error_counts)]["downgrade_percent"]
    random_max_error = random_results["by_error_count"][max(error_counts)]["downgrade_percent"]
    
    # Search vs Random improvements
    search_vs_random_single = ((search_single_error - random_single_error) / random_single_error) * 100 if random_single_error > 0 else 0
    search_vs_random_max = ((search_max_error - random_max_error) / random_max_error) * 100 if random_max_error > 0 else 0
    
    # Create summary text
    summary_lines = [
        f"Impact of Example Selection Method on Error Detection:",
        f"• Single Error Detection: Search improves over Random by {search_vs_random_single:.1f}%",
        f"• Multiple Error Detection ({max(error_counts)} errors): Search improves over Random by {search_vs_random_max:.1f}%",
    ]
    
    # Add random_then_sort comparison if available
    if random_then_sort_results:
        random_then_sort_single = random_then_sort_results["by_error_count"][1]["downgrade_percent"]
        random_then_sort_max = random_then_sort_results["by_error_count"][max(error_counts)]["downgrade_percent"]
        
        # Random-then-Sort vs Random improvements
        random_then_sort_vs_random_single = ((random_then_sort_single - random_single_error) / random_single_error) * 100 if random_single_error > 0 else 0
        random_then_sort_vs_random_max = ((random_then_sort_max - random_max_error) / random_max_error) * 100 if random_max_error > 0 else 0
        
        summary_lines.append(f"• Random-then-Sort improves over Random for Single Error: {random_then_sort_vs_random_single:.1f}%")
        summary_lines.append(f"• Random-then-Sort improves over Random for {max(error_counts)} Errors: {random_then_sort_vs_random_max:.1f}%")
    
    summary_text = "\n".join(summary_lines)
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=13, 
               bbox={"facecolor":"lightyellow", "alpha":0.9, "pad":10, "edgecolor":"orange"})
    
    # Add overall title and adjust layout
    plt.suptitle(f"Comparison of Example Selection Methods\nfor {error_level.capitalize()}-Level Error Detection", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze QE sensitivity to multiple errors")
    parser.add_argument("--source", required=True, help="Source corpus file path")
    parser.add_argument("--target", required=True, help="Target corpus file path")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--error-level", choices=["word", "character"], default="word", 
                       help="Level at which to introduce errors")
    parser.add_argument("--error-types", nargs="+", choices=["replace", "insert", "delete", "swap"],
                       default=["replace", "insert", "delete", "swap"], help="Types of errors to test")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum number of errors to introduce")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples for quality estimation")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use")
    parser.add_argument("--output-prefix", default="error_analysis", help="Output prefix for result charts")
    
    args = parser.parse_args()
    
    # Initialize corpus and estimator
    corpus = PairCorpus(source_path=args.source, target_path=args.target)
    
    # Create a combined estimator with multiple measures
    estimator = QualityEstimator(
        similarity_measures=[
            NGramSimilarity(max_n=3),
            WordOverlapSimilarity(),
            TfidfCosineSimilarity(min_n=1, max_n=3)
        ]
    )
    
    # Run the analysis
    logger.info(f"Analyzing sensitivity to {args.error_level}-level errors (1-{args.max_errors}) on {args.samples} samples using {args.workers} workers")
    results = analyze_multiple_error_sensitivity(
        corpus=corpus,
        num_samples=args.samples,
        error_level=args.error_level,
        error_types=args.error_types,
        max_errors=args.max_errors,
        estimator=estimator,
        example_size=args.examples,
        num_workers=args.workers
    )
    
    # Print summary results
    print(f"\n{args.error_level.capitalize()}-Level Error Analysis Results:")
    print("\nDowngrade Percentage by Error Count:")
    for error_count in range(1, args.max_errors + 1):
        ec_stats = results["by_error_count"][error_count]
        print(f"  {error_count} error{'s' if error_count > 1 else ''}: {ec_stats['downgrade_percent']:.1f}% downgrade, "
              f"average drop {ec_stats['average_quality_drop']:.4f}")
    
    print("\nBy Error Type:")
    for error_type in args.error_types:
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
    
    # Create visualizations
    logger.info("Creating visualizations...")
    figs = visualize_multiple_error_results(results, args.error_level)
    
    # Save figures
    for i, fig in enumerate(figs):
        output_file = f"{args.output_prefix}_{args.error_level}_{i+1}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    logger.info("Analysis complete!") 