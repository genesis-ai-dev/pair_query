import os
import json
import random
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from difflib import SequenceMatcher

import openai
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

import extract

class Translator:
    """Handles translation using OpenAI API"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize translator with model and API key
        
        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (uses env var OPENAI_API_KEY if None)
        """
        self.model_name = model_name
        self.system_prompt = "You are a translator. Translate the given text to the target language. Respond only with the translation, no other text. Translate into {language}."
        self.client = openai.OpenAI(api_key="sk-proj-")

    def translate_text(self, prompt: str, language: str) -> str:
        """
        Translate text using the OpenAI model
        
        Args:
            prompt: Text to translate with instructions
            
        Returns:
            Translated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt.format(language=language)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent translations
            )
            content = response.choices[0].message.content
            print("Prompt: ", prompt)
            print("generated content: ", content.strip())
            return content.strip() if content else ""
        except Exception as e:
            print(f"Translation error: {e}")
            return ""


def calculate_similarity(prediction: str, reference: str) -> float:
    """
    Calculate similarity between prediction and reference using multiple metrics
    
    Args:
        prediction: Generated translation
        reference: Ground truth translation
        
    Returns:
        Similarity score between 0 and 1
    """
    # If either string is empty, return 0
    if not prediction or not reference:
        return 0.0
    
    # Simple tokenization by splitting on whitespace and punctuation
    def tokenize(text):
        # Convert to lowercase and split by non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    # Get tokens
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    # Handle edge case where prediction has no tokens
    if not pred_tokens:
        return 0.0
        
    # 1. Sequence similarity using difflib
    sequence_sim = SequenceMatcher(None, prediction.lower(), reference.lower()).ratio()
    
    # 2. Word overlap (Jaccard similarity)
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    if not pred_set or not ref_set:
        jaccard_sim = 0.0
    else:
        jaccard_sim = len(pred_set.intersection(ref_set)) / len(pred_set.union(ref_set))
    
    # 3. Word order similarity
    # Create TF-IDF vectors and calculate cosine similarity
    try:
        vectorizer = TfidfVectorizer(lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([" ".join(pred_tokens), " ".join(ref_tokens)])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        # Fall back if vectorization fails
        cosine_sim = sequence_sim
    
    # Weighted combination of all similarity metrics
    # 30% sequence, 30% jaccard, 40% word order
    return 0.3 * sequence_sim + 0.3 * jaccard_sim + 0.4 * cosine_sim


def create_graphs(results: List[Dict], source_lang: str, target_lang: str, timestamp: str) -> Tuple[Figure, Figure, Figure]:
    """
    Create line, smoothed line, and bar graphs from benchmark results
    
    Args:
        results: List of test results
        source_lang: Source language name
        target_lang: Target language name
        timestamp: Timestamp string for titles
        
    Returns:
        Tuple of (line_graph, smoothed_line_graph, bar_graph) figure objects
    """
    # Sort results by index for better visualization
    sorted_results = sorted(results, key=lambda r: r["index"])
    indices = [r["index"] for r in sorted_results]
    with_examples = [r["similarity_with_examples"] for r in sorted_results]
    without_examples = [r["similarity_without_examples"] for r in sorted_results]
    
    # Calculate averages
    avg_with = np.mean(with_examples)
    avg_without = np.mean(without_examples)
    
    # Line graph
    line_fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(range(len(indices)), with_examples, 'o-', color='#2E8B57', linewidth=2, label='With Examples')
    ax1.plot(range(len(indices)), without_examples, 'o-', color='#4169E1', linewidth=2, label='Without Examples')
    
    # Add horizontal lines for averages
    ax1.axhline(y=avg_with, color='#2E8B57', linestyle='--', alpha=0.7, label=f'Avg with: {avg_with:.4f}')
    ax1.axhline(y=avg_without, color='#4169E1', linestyle='--', alpha=0.7, label=f'Avg without: {avg_without:.4f}')
    
    ax1.set_xticks(range(len(indices)))
    ax1.set_xticklabels([str(idx) for idx in indices], rotation=45)
    ax1.set_xlabel('Test Case Index', fontsize=12)
    ax1.set_ylabel('Similarity Score', fontsize=12)
    ax1.set_title(f'Translation Performance: {source_lang} → {target_lang}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    line_fig.tight_layout()
    
    # Smoothed line graph
    smooth_fig, ax_smooth = plt.subplots(figsize=(12, 6))
    
    # Apply smoothing with moving average (window size 3 or smaller if fewer points)
    window_size = min(3, len(indices))
    
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            window = data[start_idx:end_idx]
            smoothed.append(sum(window) / len(window))
        return smoothed
    
    smoothed_with = moving_average(with_examples, window_size)
    smoothed_without = moving_average(without_examples, window_size)
    
    ax_smooth.plot(range(len(indices)), smoothed_with, '-', color='#2E8B57', linewidth=3, label='With Examples (Smoothed)')
    ax_smooth.plot(range(len(indices)), smoothed_without, '-', color='#4169E1', linewidth=3, label='Without Examples (Smoothed)')
    ax_smooth.plot(range(len(indices)), with_examples, 'o', color='#2E8B57', alpha=0.4, markersize=5)
    ax_smooth.plot(range(len(indices)), without_examples, 'o', color='#4169E1', alpha=0.4, markersize=5)
    
    # Add horizontal lines for averages
    ax_smooth.axhline(y=avg_with, color='#2E8B57', linestyle='--', alpha=0.7, label=f'Avg with: {avg_with:.4f}')
    ax_smooth.axhline(y=avg_without, color='#4169E1', linestyle='--', alpha=0.7, label=f'Avg without: {avg_without:.4f}')
    
    ax_smooth.set_xticks(range(len(indices)))
    ax_smooth.set_xticklabels([str(idx) for idx in indices], rotation=45)
    ax_smooth.set_xlabel('Test Case Index', fontsize=12)
    ax_smooth.set_ylabel('Similarity Score', fontsize=12)
    ax_smooth.set_title(f'Smoothed Translation Performance: {source_lang} → {target_lang}', fontsize=14, fontweight='bold')
    ax_smooth.set_ylim(0, 1.0)
    ax_smooth.legend(loc='lower right')
    ax_smooth.grid(True, linestyle='--', alpha=0.7)
    smooth_fig.tight_layout()
    
    # Bar graph
    bar_fig, ax2 = plt.subplots(figsize=(8, 6))
    labels = ['With Examples', 'Without Examples']
    values = [avg_with, avg_without]
    colors = ['#2E8B57', '#4169E1']
    
    bars = ax2.bar(labels, values, color=colors, width=0.6)
    ax2.set_ylim(0, 1.0)
    ax2.set_title(f'Average Translation Similarity: {source_lang} → {target_lang}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Similarity Score', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels and improvement percentage
    improvement = avg_with - avg_without
    improvement_pct = (improvement / max(0.0001, avg_without)) * 100
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    
    # Add improvement text
    ax2.text(0.5, 0.1, 
             f'Improvement: {improvement:.4f} ({improvement_pct:.1f}%)',
             ha='center', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    bar_fig.tight_layout()
    
    return line_fig, smooth_fig, bar_fig


def create_progressive_chart(results_by_examples: Dict[int, List[Dict]], source_lang: str, target_lang: str, timestamp: str) -> Figure:
    """
    Create a line chart showing how performance progresses as more samples are processed
    for each example count.
    
    Args:
        results_by_examples: Dictionary mapping example counts to lists of result dictionaries
        source_lang: Source language name
        target_lang: Target language name
        timestamp: Timestamp string for titles
        
    Returns:
        Figure object with the progressive performance chart
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For each example count, create a line showing progressive average
    example_counts = sorted(results_by_examples.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(example_counts)))  # Colormap for multiple lines
    
    max_samples = max(len(results) for results in results_by_examples.values())
    
    # Find sample indices that are common across all example counts
    # This assumes the indices are the same across all example counts
    first_example_results = list(results_by_examples.values())[0]
    indices = [r["index"] for r in first_example_results]
    
    for i, count in enumerate(example_counts):
        results = results_by_examples[count]
        
        # Ensure results are sorted by the order they were processed
        results = sorted(results, key=lambda r: indices.index(r["index"]))
        
        # Calculate progressive averages (how the average evolves as more samples are processed)
        progressive_avg = []
        sum_so_far = 0
        
        for j, result in enumerate(results):
            sum_so_far += result["similarity"]
            current_avg = sum_so_far / (j + 1)
            progressive_avg.append(current_avg)
        
        # Apply adaptive smoothing - window size scales with number of samples
        window_size = max(3, len(progressive_avg) // 5)  # At least 3, or 20% of sample count
        window_size = min(window_size, 5)  # Cap at 5 to avoid over-smoothing
        
        def adaptive_smooth(data, window):
            padded = [data[0]] * (window // 2) + data + [data[-1]] * (window // 2)
            smoothed = []
            for i in range(len(data)):
                start = i
                end = i + window
                smoothed.append(sum(padded[start:end]) / window)
            return smoothed
        
        # Only smooth if we have enough points
        if len(progressive_avg) > window_size:
            smoothed_avg = adaptive_smooth(progressive_avg, window_size)
        else:
            smoothed_avg = progressive_avg
        
        # Plot both the raw and smoothed lines
        sample_numbers = list(range(1, len(progressive_avg) + 1))
        ax.plot(sample_numbers, progressive_avg, 'o-', alpha=0.3, color=colors[i], 
                markersize=4, linewidth=1)
        ax.plot(sample_numbers, smoothed_avg, '-', color=colors[i], linewidth=2.5, 
                label=f'{count} examples (avg: {progressive_avg[-1]:.4f})')
    
    # Add horizontal grid lines for reference
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Label axes and add title
    ax.set_xlabel('Number of Samples Processed', fontsize=12)
    ax.set_ylabel('Cumulative Average Similarity', fontsize=12)
    ax.set_title(f'Progressive Translation Performance: {source_lang} → {target_lang}', 
                fontsize=14, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Set x-axis ticks
    ax.set_xticks(list(range(1, max_samples + 1)))
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Add final performance annotations at the end of each line
    for i, count in enumerate(example_counts):
        results = results_by_examples[count]
        final_avg = sum(r["similarity"] for r in results) / len(results)
        ax.annotate(f'{final_avg:.4f}', 
                   (len(results), final_avg),
                   textcoords="offset points", 
                   xytext=(5, 0), 
                   ha='left')
    
    fig.tight_layout()
    return fig


def run_benchmark(
    corpus_path_source: str, 
    corpus_path_target: str, 
    num_examples: List[int] = [0, 5], 
    test_indices: Optional[List[int]] = None, 
    num_tests: int = 5, 
    output_dir: str = "benchmark_results",
    language: str = "Kosraean"
) -> Dict[str, Any]:
    """
    Run a translation benchmark comparing few-shot vs zero-shot performance.
    
    Args:
        corpus_path_source: Path to source language corpus
        corpus_path_target: Path to target language corpus
        num_examples: List of example counts to include in few-shot prompts (0 means zero-shot)
        test_indices: Specific indices to test (if None, selects random samples)
        num_tests: Number of tests to run if test_indices is None
        output_dir: Directory to save results and graphs
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract language names from file paths for labels
    try:
        source_lang = os.path.basename(corpus_path_source).split('-')[0]
        target_lang = os.path.basename(corpus_path_target).split('-')[0]
    except (IndexError, AttributeError):
        source_lang = "Source"
        target_lang = "Target"
    
    try:
        # Load corpus
        corpus = extract.PairCorpus(corpus_path_source, corpus_path_target)
        translator = Translator()
        
        # Select test indices if not provided
        if test_indices is None:
            corpus_length = len(corpus.source_lines)
            # Ensure we have enough lines to select from
            if corpus_length < num_tests:
                raise ValueError(f"Corpus only has {corpus_length} lines, cannot select {num_tests} test cases")
                
            max_idx = min(corpus_length - 1, 26000)
            min_idx = min(100, max(10, max_idx // 4))  # Start at least min(100, 10) lines in
            test_indices = random.sample(range(min_idx, max_idx), num_tests)
        
        # Ensure num_examples is a list
        if isinstance(num_examples, int):
            num_examples = [num_examples]
        
        # Make sure 0 is included for zero-shot comparison
        if 0 not in num_examples:
            num_examples = [0] + list(num_examples)
            
        all_results = {}
        zero_shot_results = []
        zero_shot_done = False
        
        for example_count in sorted(num_examples):
            print(f"\nTesting with {example_count} examples...")
            results = []
            
            for i, idx in enumerate(test_indices):
                print(f"Processing test case {i+1}/{len(test_indices)} (line {idx})...")
                
                # Get prompt with the specified number of examples
                prompt, ground_truth = corpus.format_benchmark_examples(
                    index_line=idx, 
                    top_n=example_count
                )
                
                # Translate with the current prompt
                translation = translator.translate_text(prompt, language)
                print("Ground truth: ", ground_truth)
                similarity = calculate_similarity(translation, ground_truth)
                
                # Store the result
                result = {
                    "index": idx,
                    "source": corpus.source_lines[idx].strip(),
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "translation": translation,
                    "similarity": similarity
                }
                
                # Store zero-shot results separately for comparison
                if example_count == 0:
                    zero_shot_results.append(result)
                    print(f"  - Zero-shot similarity: {similarity:.4f}")
                else:
                    # Compare with zero-shot if we have the results
                    if zero_shot_done:
                        zero_shot_sim = next((r["similarity"] for r in zero_shot_results if r["index"] == idx), 0)
                        improvement = similarity - zero_shot_sim
                        result["improvement"] = improvement
                        print(f"  - With {example_count} examples: {similarity:.4f}")
                        print(f"  - Zero-shot baseline: {zero_shot_sim:.4f}")
                        print(f"  - Improvement: {improvement:.4f}")
                
                results.append(result)
            
            # Calculate average similarity
            avg_similarity = sum(r["similarity"] for r in results) / len(results)
            
            # For non-zero-shot, calculate improvement over zero-shot
            if example_count > 0 and zero_shot_done:
                improvements = []
                for r in results:
                    idx = r["index"]
                    zero_shot_sim = next((zr["similarity"] for zr in zero_shot_results if zr["index"] == idx), 0)
                    improvements.append(r["similarity"] - zero_shot_sim)
                avg_improvement = sum(improvements) / len(improvements)
            else:
                avg_improvement = 0
                
            summary = {
                "num_examples": example_count,
                "num_tests": len(test_indices),
                "avg_similarity": avg_similarity,
                "detailed_results": results
            }
            
            if example_count > 0 and zero_shot_done:
                summary["avg_improvement"] = avg_improvement
                
            print(f"\nSummary for {example_count} examples:")
            print(f"Average similarity: {avg_similarity:.4f}")
            if example_count > 0 and zero_shot_done:
                print(f"Average improvement over zero-shot: {avg_improvement:.4f}")
                zero_shot_avg = all_results[0]["avg_similarity"]
                improvement_pct = (avg_improvement / max(0.0001, zero_shot_avg)) * 100
                print(f"Improvement percentage: {improvement_pct:.1f}%")
            
            # Save results to file
            results_file = f'{output_dir}/benchmark_results_{example_count}ex_{timestamp}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                # Remove detailed results to keep file size manageable
                summary_for_file = {
                    "num_examples": example_count,
                    "num_tests": summary["num_tests"],
                    "avg_similarity": summary["avg_similarity"],
                    "test_indices": test_indices,
                    "source_path": corpus_path_source,
                    "target_path": corpus_path_target,
                    "model": translator.model_name,
                    "timestamp": timestamp
                }
                
                if example_count > 0 and zero_shot_done:
                    summary_for_file["avg_improvement"] = avg_improvement
                    summary_for_file["improvement_percentage"] = improvement_pct
                    
                json.dump(summary_for_file, f, ensure_ascii=False, indent=2)
                
            all_results[example_count] = summary
            
            # Mark zero-shot as done after processing
            if example_count == 0:
                zero_shot_done = True
        
        # Create comparison graphs only if we have zero-shot results
        if 0 in all_results and len(num_examples) > 1:
            # Prepare for bar and line graphs comparing all example counts
            example_counts = sorted([k for k in all_results.keys() if k > 0])
            few_shot_scores = [all_results[count]["avg_similarity"] for count in example_counts]
            zero_shot_score = all_results[0]["avg_similarity"]
            
            # Bar graph comparing zero-shot with all few-shot variants
            bar_fig, ax_bar = plt.subplots(figsize=(10, 6))
            labels = ['Zero-shot'] + [f'{count} examples' for count in example_counts]
            values = [zero_shot_score] + few_shot_scores
            colors = ['#4169E1'] + ['#2E8B57'] * len(example_counts)
            
            bars = ax_bar.bar(labels, values, color=colors, width=0.6)
            ax_bar.set_ylim(0, 1.0)
            ax_bar.set_title(f'Translation Performance: {source_lang} → {target_lang}', fontsize=14, fontweight='bold')
            ax_bar.set_ylabel('Average Similarity Score', fontsize=12)
            ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=11)
            
            bar_fig.tight_layout()
            bar_fig.savefig(f'{output_dir}/bar_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            
            # Line graph showing effect of example count
            line_fig, ax_line = plt.subplots(figsize=(10, 6))
            all_counts = [0] + example_counts
            all_scores = [zero_shot_score] + few_shot_scores
            
            ax_line.plot(all_counts, all_scores, 'o-', color='#2E8B57', linewidth=2)
            
            ax_line.set_xlabel('Number of Examples', fontsize=12)
            ax_line.set_ylabel('Average Similarity Score', fontsize=12)
            ax_line.set_title(f'Effect of Example Count on Translation: {source_lang} → {target_lang}', fontsize=14, fontweight='bold')
            ax_line.set_xticks(all_counts)
            ax_line.set_ylim(0, 1.0)
            ax_line.grid(True, linestyle='--', alpha=0.7)
            
            # Add data labels
            for i, count in enumerate(all_counts):
                score = all_scores[i]
                ax_line.annotate(f'{score:.4f}', 
                                (count, score),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
            
            line_fig.tight_layout()
            line_fig.savefig(f'{output_dir}/example_count_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            
            # Add the progressive performance chart
            progressive_results = {
                count: all_results[count]["detailed_results"] 
                for count in all_results.keys()
            }
            
            progressive_fig = create_progressive_chart(
                progressive_results, 
                source_lang, 
                target_lang, 
                timestamp
            )
            
            progressive_fig.savefig(f'{output_dir}/progressive_performance_{timestamp}.png', 
                                  dpi=300, bbox_inches='tight')
        
        print(f"Results and graphs saved to {output_dir}")
        
        return all_results if len(num_examples) > 1 else list(all_results.values())[0]
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        raise


if __name__ == "__main__":
    results = run_benchmark(
        corpus_path_source="pair_query/corpus/eng-engULB.txt", 
        corpus_path_target="pair_query/corpus/kos-kos.txt", 
        num_examples=[0, 5, 10],  # Include zero-shot (0) and various few-shot counts
        num_tests=5,
        language="Kosraean"
    )
    