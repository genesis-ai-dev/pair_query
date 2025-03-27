from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Set
from pathlib import Path
from functools import lru_cache
from tqdm.auto import tqdm  # type: ignore
import logging

# Import similarity measures from the new module
from pq.similarity_measures import (
    SimilarityMeasure, 
    NGramSimilarity, 
    WordOverlapSimilarity,
    LongestSubstringSimilarity,
    CombinedSimilarity,
    LengthSimilarity,
    WordEditDistanceSimilarity,
    TfidfCosineSimilarity
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    line_idx: int
    similarity: float
    source_line: str
    target_line: str

class PairCorpus:
    """Base class for handling parallel text corpora."""
    
    def __init__(self, source_path: Union[str, Path], target_path: Union[str, Path]):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)

        with open(self.source_path, encoding="utf-8") as f:
            self.source_lines = f.readlines()

        with open(self.target_path, encoding="utf-8") as f:
            self.target_lines = f.readlines()
            
        self._vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")

    def get_pairs(self, line_number: int) -> Tuple[str, str]:
        """Get a source-target pair by line number."""
        return self.source_lines[line_number], self.target_lines[line_number]
        
    def search(self, 
               index_line: int, 
               search_term: Optional[str] = None, 
               previous_only: bool = False, 
               top_n: int = 5) -> List[SearchResult]:
        """
        Search for similar lines to the one at index_line or for a specific search term.
        """
        search_term = search_term or self.source_lines[index_line].strip()
            
        search_range = range(0, index_line) if previous_only else range(len(self.source_lines))

        # Handle the case where search_range is empty (when previous_only=True and index_line=0)
        if not search_range:
            raise ValueError("Cannot search previous lines when index_line is 0 and previous_only=True")
            
        lines_to_search = [self.source_lines[i].strip() for i in search_range]
        
        all_texts = lines_to_search + [search_term]
        tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        
        search_term_vector = tfidf_matrix[-1:]
        similarities = cosine_similarity(search_term_vector, tfidf_matrix[:-1]).flatten()
        
        results = [
            SearchResult(
                line_idx=i,
                similarity=float(similarity),
                source_line=self.source_lines[i],
                target_line=self.target_lines[i]
            )
            for i, similarity in enumerate(similarities) if i in search_range
        ]
        
        return sorted(results, key=lambda x: x.similarity, reverse=True)[:top_n]


class TextDegradation:
    """Class for text degradation methods."""
    
    @staticmethod
    def generate_random_word(length: int) -> str:
        """Generate a random word of given length."""
        word_len = max(1, int(random.gauss(length, 2)))
        return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') 
                     for _ in range(word_len))

    @staticmethod
    def degrade_random(text: str, level: float) -> str:
        """Degrade text randomly based on level (0.0-1.0)."""
        if not text or level <= 0:
            return text
            
        words = text.split()
        
        # Complete random text for level 1.0
        if level >= 1.0:
            return ' '.join(
                TextDegradation.generate_random_word(len(word))
                for word in words
            )
        
        # Partial degradation
        text_len = len(words)
        num_to_replace = int(text_len * level)
        
        # Maybe shuffle words (for level > 0.5)
        if level > 0.5:
            shuffle_count = int((level - 0.5) * text_len)
            positions = random.sample(range(text_len), min(shuffle_count, text_len))
            words_to_shuffle = [words[i] for i in positions]
            random.shuffle(words_to_shuffle)
            for i, pos in enumerate(positions):
                words[pos] = words_to_shuffle[i]
        
        # Replace words with random ones
        positions = random.sample(range(text_len), min(num_to_replace, text_len))
        for pos in positions:
            words[pos] = TextDegradation.generate_random_word(len(words[pos]))
        
        return ' '.join(words)
    
    @classmethod
    def degrade_transition(cls, text: str, level: float, corpus: PairCorpus, original_idx: int) -> str:
        """Transition text toward another text in corpus based on level (0.0-1.0)."""
        if level <= 0:
            return text
            
        valid_indices = [i for i in range(len(corpus.target_lines)) 
                        if i != original_idx and corpus.target_lines[i].strip()]
            
        if not valid_indices:
            return text
        
        # Pick a target text deterministically
        random.seed(hash(text) % 10000)
        target_text = corpus.target_lines[random.choice(valid_indices)].strip()
        random.seed()
        
        if level >= 1.0:
            return target_text
        
        # Partial transition
        words1 = text.split()
        words2 = target_text.split()
        
        # Determine final word count
        final_len = int(len(words1) * (1 - level) + len(words2) * level)
        result = words1.copy()
        
        # Adjust length
        if final_len < len(result):
            # Remove words
            for _ in range(len(result) - final_len):
                if result:
                    result.pop(random.randrange(len(result)))
        elif final_len > len(result):
            # Add words from target
            for _ in range(final_len - len(result)):
                if words2:
                    word = random.choice(words2)
                    pos = random.randint(0, len(result))
                    result.insert(pos, word)
        
        # Replace words with target words
        replace_count = int(len(result) * level)
        positions = random.sample(range(len(result)), min(replace_count, len(result)))
        
        for pos in positions:
            if words2:
                result[pos] = random.choice(words2)
        
        return ' '.join(result)


class QualityEstimator:
    """Class for translation quality estimation."""
    
    def __init__(self, 
                similarity_measures: Optional[Union[SimilarityMeasure, List[SimilarityMeasure]]] = None,
                weights: Optional[List[float]] = None,
                combination_mode: str = "multiply"):
        """
        Initialize with optional similarity measures and combination settings.
        
        Args:
            similarity_measures: Single measure or list of measures to use
            weights: Optional weights for combining correlations (must match number of measures)
            combination_mode: How to combine correlations - "multiply" or "average"
        """
        # Handle single measure or default
        if similarity_measures is None:
            self.similarity_measures = [NGramSimilarity()]  # type: List[SimilarityMeasure]
        elif isinstance(similarity_measures, SimilarityMeasure):
            self.similarity_measures = [similarity_measures]  # type: List[SimilarityMeasure]
        else:
            self.similarity_measures = similarity_measures  # type: List[SimilarityMeasure]
            
        # Validate combination mode
        if combination_mode not in ["multiply", "average"]:
            raise ValueError("Combination mode must be either 'multiply' or 'average'")
        self.combination_mode = combination_mode
        
        # Set and normalize weights
        if weights:
            if len(weights) != len(self.similarity_measures):
                raise ValueError("Number of weights must match number of similarity measures")
            total = sum(weights)
            self.weights = [w / total for w in weights] if total > 0 else [1.0 / len(self.similarity_measures)] * len(self.similarity_measures)
        else:
            self.weights = [1.0 / len(self.similarity_measures)] * len(self.similarity_measures)
    
    def calculate_similarity(self, text1: str, text2: str, measure_index: int = 0) -> float:
        """Calculate similarity between two texts using the specified measure."""
        return self.similarity_measures[measure_index].calculate_similarity(text1, text2)
    
    def evaluate_translation(self, 
                           source: str, 
                           translation: str, 
                           corpus: PairCorpus,
                           sample_size: int = 10,
                           example_selection: str = "search",
                           previous_only: bool = False,
                           source_index: int = 0) -> float:
        """
        Evaluate translation quality using reference corpus.
        
        For each similarity measure:
        1. Calculate correlation between source and target similarities
        2. Combine correlations using the specified combination mode
        
        Args:
            source: Source text
            translation: Translation to evaluate
            corpus: Reference corpus
            sample_size: Number of examples to use
            example_selection: Method to select examples ('search', 'random', or 'random_then_sort')
            previous_only: Whether to only use previous examples
            source_index: Index of the source text in the corpus (if known)
        
        Returns a score between 0 and 1.
        """
        # Get reference examples based on selection method
        if example_selection == "random":
            # Select random examples from corpus
            valid_indices = [i for i in range(len(corpus.source_lines)) 
                           if len(corpus.source_lines[i].strip()) > 0 
                           and len(corpus.target_lines[i].strip()) > 0]
            
            if len(valid_indices) < sample_size:
                sample_size = len(valid_indices)
                
            if sample_size < 2:
                return 0.0
                
            sample_indices = random.sample(valid_indices, sample_size)
            
            # Create SearchResult objects to match the interface expected below
            examples = [
                SearchResult(
                    line_idx=idx,
                    similarity=0.0,  # Not used for random selection
                    source_line=corpus.source_lines[idx],
                    target_line=corpus.target_lines[idx]
                )
                for idx in sample_indices
            ]
        else:
            # Use similarity search (default behavior)
            # If source_index was not provided, try to find it in the corpus
            if source_index == 0:
                for i, line in enumerate(corpus.source_lines):
                    if line.strip() == source.strip():
                        source_index = i
                        break
            
            # Perform the search with the identified index
            examples = corpus.search(source_index, search_term=source, previous_only=previous_only, top_n=sample_size)
            
        if len(examples) < 2:
            return 0.0
            
        measure_correlations = []
        
        # Calculate correlation for each similarity measure
        for measure_idx, measure in enumerate(self.similarity_measures):
            source_similarities = []
            target_similarities = []
            
            for ex in examples:
                source_sim = measure.calculate_similarity(source, ex.source_line)
                target_sim = measure.calculate_similarity(translation, ex.target_line)
                
                source_similarities.append(source_sim)
                target_similarities.append(target_sim)
            
            # Calculate correlation for this measure
            correlation = np.corrcoef(source_similarities, target_similarities)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else max(0.0, correlation)
            measure_correlations.append(correlation)
        
        # Combine correlations according to the chosen mode
        if self.combination_mode == "multiply":
            # Joint probability (weighted product)
            result = 1.0
            for corr, weight in zip(measure_correlations, self.weights):
                result *= corr ** weight
            return result
            
        else:  # combination_mode == "average"
            # Weighted average
            return sum(corr * weight for corr, weight in zip(measure_correlations, self.weights))


class Benchmarker:
    """Class for benchmarking translation quality."""
    
    @dataclass
    class TestResult:
        source: str
        reference: str
        translation: str
        score: float
    
    @staticmethod
    def format_examples(corpus: PairCorpus, index_line: int, example_count: int = 5) -> str:
        """Format few-shot examples for translation."""
        examples = corpus.search(index_line, previous_only=True, top_n=example_count)
        
        prompt = "Translate the following text. Here are some examples:\n\n"
        
        for i, ex in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Source: {ex.source_line.strip()}\n"
            prompt += f"Translation: {ex.target_line.strip()}\n\n"
        
        prompt += f"Now translate this:\n{corpus.source_lines[index_line].strip()}"
        
        return prompt
    
    @classmethod
    def test_quality_metric(cls, 
                          corpus: PairCorpus,
                          num_tests: int = 5,
                          degradation_levels: int = 5,
                          mode: str = "random",
                          example_size: int = 10,
                          quality_estimator: Optional[QualityEstimator] = None,
                          example_selection: str = "random") -> Dict:
        """Test quality estimation by comparing with degraded translations.
        
        Args:
            corpus: The parallel corpus to use
            num_tests: Number of sentence pairs to test
            degradation_levels: Number of degradation levels to test
            mode: Degradation mode ("random" or "transition")
            example_size: Number of examples to use for quality estimation
            quality_estimator: Optional custom quality estimator
        """
        
        # Find valid test examples
        valid_indices = [i for i in range(len(corpus.source_lines)) 
                       if len(corpus.source_lines[i].strip()) > 20 
                       and len(corpus.target_lines[i].strip()) > 20]
        
        if len(valid_indices) < num_tests:
            raise ValueError(f"Not enough valid sentence pairs. Need {num_tests}, found {len(valid_indices)}")
        
        test_indices = random.sample(valid_indices, num_tests)
        results = []
        
        # Use provided estimator or create a default one
        estimator = quality_estimator or QualityEstimator()
        
        for idx in tqdm(test_indices, desc="Testing quality metric", unit="pair"):
            source = corpus.source_lines[idx].strip()
            reference = corpus.target_lines[idx].strip()
            
            # Evaluate original quality
            baseline = estimator.evaluate_translation(
                source, reference, corpus, sample_size=example_size, example_selection="random"
            )
            
            level_scores = []
            
            # Test different degradation levels
            for level in range(1, degradation_levels + 1):
                deg_level = level / degradation_levels
                
                if mode == "transition":
                    degraded = TextDegradation.degrade_transition(
                        reference, deg_level, corpus, idx
                    )
                else:
                    degraded = TextDegradation.degrade_random(reference, deg_level)
                
                score = estimator.evaluate_translation(
                    source, degraded, corpus, sample_size=example_size
                )
                level_scores.append(score)
            
            results.append({
                "source": source,
                "reference": reference,
                "baseline": baseline,
                "degraded_scores": level_scores
            })
        
        # Calculate metrics
        avg_baseline = np.mean([r["baseline"] for r in results])
        avg_degraded = [np.mean([r["degraded_scores"][i] for r in results]) 
                      for i in range(degradation_levels)]
        
        drop = avg_baseline - avg_degraded[-1]
        monotonic = all(avg_degraded[i] >= avg_degraded[i+1] for i in range(degradation_levels-1))
        
        # Create visualization
        fig = cls._create_quality_plot(results, avg_baseline, avg_degraded, degradation_levels)
        
        return {
            "results": results,
            "avg_baseline": avg_baseline,
            "avg_degraded": avg_degraded,
            "quality_drop": drop,
            "is_monotonic": monotonic,
            "figure": fig
        }
    
    @staticmethod
    def _create_quality_plot(results, baseline, degraded, levels, ax=None):
        """Create visualization of quality test results."""
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        # Plot individual test lines (lightly)
        for i, result in enumerate(results):
            ax.plot([0] + list(range(1, levels + 1)),
                    [result["baseline"]] + result["degraded_scores"], 
                    'o-', alpha=0.3, label=f"Test {i+1}")
        
        # Plot average line (bold)
        ax.plot([0] + list(range(1, levels + 1)), 
                [baseline] + degraded, 'bo-', linewidth=2, markersize=8, label="Average")
        
        # Add info box
        quality_drop = baseline - degraded[-1]
        is_monotonic = all(degraded[i] >= degraded[i+1] for i in range(levels-1))
        effectiveness = int(min(100, max(0, quality_drop * 100)))
        
        ax.text(0.02, 0.02, 
                f"Quality drop: {quality_drop:.2f}\n"
                f"Monotonic: {'Yes' if is_monotonic else 'No'}\n"
                f"Effectiveness: {effectiveness}%", 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Degradation Level')
        ax.set_ylabel('Quality Score')
        ax.set_title('Quality vs. Degradation')
        ax.grid(True)
        ax.set_xticks(range(0, levels + 1))
        ax.set_xticklabels(['0%'] + [f'{int(i/levels*100)}%' for i in range(1, levels + 1)])
        
        return fig


if __name__ == "__main__":
    corpus = PairCorpus(source_path="../files/corpus/eng-engULB.txt", target_path="../files/corpus/kos-kos.txt")
    
    # Create individual estimators for comparison
    word_overlap_estimator = QualityEstimator(
        similarity_measures=WordOverlapSimilarity()
    )
    
    substring_estimator = QualityEstimator(
        similarity_measures=LongestSubstringSimilarity()
    )
    
    length_estimator = QualityEstimator(
        similarity_measures=LengthSimilarity(compare_words=True)
    )
    
    # Create a combined estimator with these measures
    combined_estimator = QualityEstimator(
        similarity_measures=[
            # WordOverlapSimilarity(),
            # LongestSubstringSimilarity(),
            # LengthSimilarity(compare_words=True),
            # WordEditDistanceSimilarity(),  # word-level edit distance
            TfidfCosineSimilarity(min_n=1, max_n=20)  # TF-IDF with uni, bi, and trigrams
        ],
        combination_mode="multiply"
    )
    
    # Test with the combined estimator
    print("Testing with combined similarity measures...")
    random_result = Benchmarker.test_quality_metric(
        corpus=corpus,
        num_tests=5,
        degradation_levels=5,
        mode="transition",
        example_size=25,
        quality_estimator=combined_estimator
    )
    
    # print("\nTesting transition degradation mode...")
    # transition_result = Benchmarker.test_quality_metric(
    #     corpus=corpus,
    #     num_tests=3,
    #     degradation_levels=5,
    #     mode="transition",
    #     example_size=100,
    #     quality_estimator=combined_estimator
    # )
    
    # Display results
    print("\nResults summary:")
    print(f"Random mode - Quality drop: {random_result['quality_drop']:.2f}, "
          f"Monotonic: {random_result['is_monotonic']}")
    # print(f"Transition mode - Quality drop: {transition_result['quality_drop']:.2f}, "
    #       f"Monotonic: {transition_result['is_monotonic']}")
    
    # Show side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    Benchmarker._create_quality_plot(
        random_result["results"], 
        random_result["avg_baseline"], 
        random_result["avg_degraded"], 
        5, 
        ax=ax1
    )
    ax1.set_title("Random Degradation (Combined Measures)")
    
    # Benchmarker._create_quality_plot(
    #     transition_result["results"], 
    #     transition_result["avg_baseline"], 
    #     transition_result["avg_degraded"], 
    #     5, 
    #     ax=ax2
    # )
    # ax2.set_title("Transition Degradation (Combined Measures)")
    
    plt.tight_layout()
    plt.show()