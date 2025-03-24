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
                                           List[ErrorType], int, int]) -> SampleResults:
    """Analyze a single sample with multiple error types and counts.
    
    Args:
        sample_info: Tuple containing (source_text, reference_text, idx, estimator, corpus, 
                                       error_level, error_types, max_errors, example_size)
    
    Returns:
        Analysis results for this sample
    """
    source, reference, idx, estimator, corpus, error_level, error_types, max_errors, example_size = sample_info
    
    # Evaluate original quality once
    original_quality = estimator.evaluate_translation(source, reference, corpus, sample_size=example_size)
    
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
                source, modified, corpus, sample_size=example_size
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
    num_workers: int = 4
) -> AnalysisResults:
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
    
    Returns:
        Dictionary containing analysis results
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
    
    # Prepare data for parallel processing
    sample_data = [
        (corpus.source_lines[idx].strip(), corpus.target_lines[idx].strip(), idx, 
         estimator, corpus, error_level, error_types, max_errors, example_size)
        for idx in test_indices
    ]
    
    # Process samples in parallel
    logger.info(f"Processing {len(sample_data)} samples with {num_workers} workers")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the analyze_single_sample function to each sample
        future_to_sample = {executor.submit(analyze_single_sample, sample_info): sample_info 
                          for sample_info in sample_data}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_sample), total=len(sample_data), 
                         desc=f"Testing {error_level}-level errors", unit="sample"):
            sample_result = future.result()
            results["samples"].append(sample_result)
            
            # Update statistics based on this sample's results
            for error_count in range(1, max_errors + 1):
                if error_count not in sample_result["error_results"]:
                    continue
                    
                for test_result in sample_result["error_results"][error_count]:
                    error_type = test_result["error_type"]
                    is_downgrade = test_result["is_downgrade"]
                    quality_drop = test_result["quality_drop"]
                    
                    # Update error count statistics
                    ec_stats = results["by_error_count"][error_count]
                    ec_stats["total_tests"] += 1
                    
                    if is_downgrade:
                        ec_stats["downgrade_count"] += 1
                        ec_stats["total_quality_drop"] += quality_drop
                    
                    # Update error type statistics for this error count
                    et_stats = ec_stats["by_error_type"][error_type]
                    et_stats["count"] += 1
                    
                    if is_downgrade:
                        et_stats["downgrade_count"] += 1
                        et_stats["quality_drop"] += quality_drop
    
    # Calculate overall statistics for each error count
    for error_count in range(1, max_errors + 1):
        ec_stats = results["by_error_count"][error_count]
        
        # Calculate downgrade percentage
        if ec_stats["total_tests"] > 0:
            ec_stats["downgrade_percent"] = (ec_stats["downgrade_count"] / ec_stats["total_tests"]) * 100
        
        # Calculate average quality drop
        if ec_stats["downgrade_count"] > 0:
            ec_stats["average_quality_drop"] = ec_stats["total_quality_drop"] / ec_stats["downgrade_count"]
        
        # Calculate per-error-type statistics
        for error_type in error_types:
            et_stats = ec_stats["by_error_type"][error_type]
            
            # Downgrade percentage for this error type
            if et_stats["count"] > 0:
                downgrade_percent = (et_stats["downgrade_count"] / et_stats["count"]) * 100
                results["by_error_type"][error_type]["downgrade_percent"].append(downgrade_percent)
            else:
                results["by_error_type"][error_type]["downgrade_percent"].append(0.0)
            
            # Average quality drop for this error type
            if et_stats["downgrade_count"] > 0:
                avg_drop = et_stats["quality_drop"] / et_stats["downgrade_count"]
                results["by_error_type"][error_type]["avg_quality_drop"].append(avg_drop)
            else:
                results["by_error_type"][error_type]["avg_quality_drop"].append(0.0)
    
    return results

def visualize_multiple_error_results(results: AnalysisResults, error_level: str = "word") -> Tuple[plt.Figure, ...]:
    """Create visualizations of multiple error sensitivity results.
    
    Args:
        results: The analysis results dictionary
        error_level: The level of errors tested ("word" or "character")
        
    Returns:
        Tuple of matplotlib figures
    """
    # Use seaborn style
    sns.set_style("whitegrid")
    
    # Extract common data
    error_counts = results["error_counts"]
    error_types = list(results["by_error_type"].keys())
    
    # Figure 1: Line chart of downgrade percentage by error count and type
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Extract downgrade percentages for each error type
    for error_type in error_types:
        downgrade_percents = results["by_error_type"][error_type]["downgrade_percent"]
        ax1.plot(error_counts, downgrade_percents, 'o-', linewidth=2, markersize=8, label=f"{error_type.capitalize()}")
    
    # Calculate and plot overall average
    overall_percents = [results["by_error_count"][ec]["downgrade_percent"] for ec in error_counts]
    ax1.plot(error_counts, overall_percents, 'ko-', linewidth=3, markersize=10, label="Overall Average")
    
    # Annotations and formatting
    for i, percent in enumerate(overall_percents):
        ax1.annotate(f"{percent:.1f}%", 
                   (error_counts[i], percent + 1), 
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    ax1.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=12)
    ax1.set_ylabel('Quality Downgrade Percentage (%)', fontsize=12)
    ax1.set_title(f'Percentage of Samples with Quality Downgrade vs. Number of {error_level.capitalize()}-Level Errors', 
                fontsize=14, fontweight='bold')
    ax1.set_xticks(error_counts)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Figure 2: Line chart of average quality drop by error count and type
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Extract quality drops for each error type
    for error_type in error_types:
        quality_drops = results["by_error_type"][error_type]["avg_quality_drop"]
        ax2.plot(error_counts, quality_drops, 'o-', linewidth=2, markersize=8, label=f"{error_type.capitalize()}")
    
    # Calculate and plot overall average
    overall_drops = [results["by_error_count"][ec]["average_quality_drop"] for ec in error_counts]
    ax2.plot(error_counts, overall_drops, 'ko-', linewidth=3, markersize=10, label="Overall Average")
    
    # Annotations and formatting
    for i, drop in enumerate(overall_drops):
        ax2.annotate(f"{drop:.4f}", 
                   (error_counts[i], drop + 0.001), 
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    ax2.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=12)
    ax2.set_ylabel('Average Quality Drop', fontsize=12)
    ax2.set_title(f'Average Quality Drop vs. Number of {error_level.capitalize()}-Level Errors', 
                fontsize=14, fontweight='bold')
    ax2.set_xticks(error_counts)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Figure 3: Heatmap of downgrade percentage by error type and count
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(error_types), len(error_counts)))
    for i, error_type in enumerate(error_types):
        for j, error_count in enumerate(error_counts):
            et_stats = results["by_error_count"][error_count]["by_error_type"][error_type]
            if et_stats["count"] > 0:
                heatmap_data[i, j] = (et_stats["downgrade_count"] / et_stats["count"]) * 100
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
               xticklabels=error_counts, yticklabels=[et.capitalize() for et in error_types],
               ax=ax3, cbar_kws={'label': 'Downgrade Percentage (%)'})
    
    ax3.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=12)
    ax3.set_ylabel('Error Type', fontsize=12)
    ax3.set_title(f'Quality Downgrade Percentage by Error Type and Count ({error_level.capitalize()}-Level)', 
                fontsize=14, fontweight='bold')
    
    # Figure 4: Heatmap of average quality drop by error type and count
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(error_types), len(error_counts)))
    for i, error_type in enumerate(error_types):
        for j, error_count in enumerate(error_counts):
            et_stats = results["by_error_count"][error_count]["by_error_type"][error_type]
            if et_stats["downgrade_count"] > 0:
                heatmap_data[i, j] = et_stats["quality_drop"] / et_stats["downgrade_count"]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis",
               xticklabels=error_counts, yticklabels=[et.capitalize() for et in error_types],
               ax=ax4, cbar_kws={'label': 'Average Quality Drop'})
    
    ax4.set_xlabel(f'Number of {error_level.capitalize()}-Level Errors', fontsize=12)
    ax4.set_ylabel('Error Type', fontsize=12)
    ax4.set_title(f'Average Quality Drop by Error Type and Count ({error_level.capitalize()}-Level)', 
                fontsize=14, fontweight='bold')
    
    # Ensure figures are displayed nicely
    for fig in [fig1, fig2, fig3, fig4]:
        fig.tight_layout()
    
    return fig1, fig2, fig3, fig4

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