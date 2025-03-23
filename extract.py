from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np
import difflib
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm  # type: ignore  # Import tqdm for progress bars
    
class PairCorpus:
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path

        with open(self.source_path, encoding="utf-8", mode="r") as f:
            self.source_lines = f.readlines()

        with open(self.target_path, encoding="utf-8", mode="r") as f:
            self.target_lines = f.readlines()
            
        # Initialize vectorizer on first use
        self._vectorizer = None

    def get_pairs(self, line_number):
        return self.source_lines[line_number], self.target_lines[line_number]
        
    def search(self, index_line, search_term=None, previous_only=False, limit=None, top_n=5):
        """
        Search for similar lines to the one at index_line or for a specific search term.
        
        Args:
            index_line: Line number to use as reference
            search_term: Optional string to search for. If None, uses the content of source_lines[index_line]
            previous_only: If True, only search in lines before the index_line
            limit: Maximum number of results to return (None means no limit)
            top_n: Number of top results to return, sorted by similarity
            
        Returns:
            List of tuples (line_number, similarity_score, source_line, target_line) for matches,
            sorted by similarity score in descending order.
        """
        # If no search term is provided, use the text from the index line
        if search_term is None:
            search_term = self.source_lines[index_line].strip()
            
        # Skip empty searches
        if not search_term.strip():
            return []
            
        # Determine search range based on previous_only flag
        if previous_only:
            search_range = list(range(0, index_line))
        else:
            search_range = list(range(len(self.source_lines)))
            
        if not search_range:
            return []
            
        # Get the lines to search through
        lines_to_search = [self.source_lines[i].strip() for i in search_range]
        
        # Initialize vectorizer if needed
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
            
        # Vectorize the search term and lines
        all_texts = lines_to_search + [search_term]
        tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between search term and all lines
        search_term_vector = tfidf_matrix[-1:]  # Last row is the search term
        similarities = cosine_similarity(search_term_vector, tfidf_matrix[:-1]).flatten()
        
        # Create results with index, similarity score, source line, and target line
        results = []
        for i, similarity in enumerate(similarities):
            line_idx = search_range[i]
            results.append((
                line_idx,
                float(similarity),  # Convert from numpy type to Python float
                self.source_lines[line_idx],
                self.target_lines[line_idx]
            ))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_n limit
        results = results[:top_n]
        
        # Apply additional limit if specified
        if limit and limit < len(results):
            results = results[:limit]
                    
        return results

    def format_benchmark_examples(self, index_line, top_n=5):
        """
        Format benchmark examples for LLM translation testing.
        
        Args:
            index_line: The line number to use as the test case
            top_n: Number of similar examples to include in the few-shot prompt.
                   If top_n=0, returns a zero-shot prompt with no examples.
            
        Returns:
            tuple of (prompt, ground_truth) where prompt contains examples if top_n > 0
        """
        # Get the source text we want to translate
        source_text = self.source_lines[index_line].strip()
        # Get the ground truth translation
        ground_truth = self.target_lines[index_line].strip()
        
        # For zero-shot case (no examples)
        if top_n == 0:
            return f"Translate the following text:\n\n{source_text}", ground_truth
        
        # Get similar examples (excluding the test case itself)
        similar_examples = self.search(
            index_line=index_line, 
            previous_only=True,  # Only use previous examples to avoid data leakage
            top_n=top_n
        )
        
        # Format the few-shot prompt (with examples)
        prompt = "Translate for me. Here are some examples:\n\n"
        
        # Add the examples
        for i, (_, score, src, tgt) in enumerate(similar_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Source: {src.strip()}\n"
            prompt += f"Translation: {tgt.strip()}\n\n"
        
        # Add the target text to translate
        prompt += f"Now translate this:\n{source_text}"
        
        return prompt, ground_truth
    
    def evaluate_quality(self, source_sentence, draft_translation, example_size=10, use_search=True, max_ngram=5, weight_base=2):
        """
        Evaluate translation quality by comparing n-gram overlap patterns in source and target languages.
        
        Args:
            source_sentence: The source sentence to be translated
            draft_translation: The draft translation to evaluate
            example_size: Number of reference sentence pairs to use for correlation
            use_search: If True, uses similar sentences found via search. If False, uses random sampling.
            max_ngram: Maximum n-gram size to consider (1=unigrams, 2=bigrams, 3=trigrams, etc.)
            weight_base: Base for exponential weighting of n-grams (higher values give more weight to longer n-grams)
            
        Returns:
            float: Correlation score between source and target similarity patterns
                Higher score indicates better translation quality
        """
        # Get sample indices either through search or random sampling
        if use_search:
            # Find similar sentences using the search method
            search_results = self.search(
                index_line=0,  # Dummy index (not used since we provide search_term)
                search_term=source_sentence,
                previous_only=False,
                top_n=example_size
            )
            sample_indices = [idx for idx, _, _, _ in search_results]
        else:
            # Random sampling approach
            valid_indices = [i for i in range(len(self.source_lines)) 
                             if self.source_lines[i].strip() and self.target_lines[i].strip()]
            sample_indices = random.sample(valid_indices, min(example_size, len(valid_indices)))
        
        # Helper function to generate n-grams of any size
        def get_ngrams(text, n):
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        # Helper function to calculate weighted n-gram similarity
        def calculate_similarity(text1, text2, max_n):
            # Use exponential weighting: weight = base^n
            # This gives exponentially higher weights to longer n-grams
            weights = {n: weight_base**n for n in range(1, max_n+1)}
            total_weight = sum(weights.values())
            
            weighted_sim = 0
            for n in range(1, max_n+1):
                # Skip if either text is too short for this n-gram size
                if len(text1.split()) < n or len(text2.split()) < n:
                    continue
                    
                ngrams1 = set(get_ngrams(text1, n))
                ngrams2 = set(get_ngrams(text2, n))
                
                # Jaccard similarity for this n-gram level
                overlap = len(ngrams1 & ngrams2)
                union = len(ngrams1 | ngrams2)
                
                if union > 0:
                    # Square the individual similarity scores to enhance the impact of matches
                    sim = (overlap / union) 
                    weighted_sim += sim * (weights[n] / total_weight)
            
            return weighted_sim
        
        # Calculate similarity vectors
        source_similarities = []
        target_similarities = []
        
        for idx in sample_indices:
            # Get reference texts
            reference_source = self.source_lines[idx].strip()
            reference_target = self.target_lines[idx].strip()
            
            # Calculate similarities with weighted n-grams
            source_similarity = calculate_similarity(source_sentence, reference_source, max_ngram)
            target_similarity = calculate_similarity(draft_translation, reference_target, max_ngram)
            
            source_similarities.append(source_similarity)
            target_similarities.append(target_similarity)
        
        # Calculate correlation between source and target similarity patterns
        if len(sample_indices) < 2:
            return 0.0  # Not enough samples for correlation
        
        correlation = np.corrcoef(source_similarities, target_similarities)[0, 1]
        
        # Handle potential NaN values (when one array has no variance)
        if np.isnan(correlation):
            return 0.0
            
        return correlation
    
    def test_quality_metric(self, num_tests=5, num_pairs=10, degradation_levels=5):
        """
        Test the quality estimation method by comparing original translations with 
        increasingly degraded versions.
        
        Args:
            num_tests: Number of sentence pairs to test
            num_pairs: Number of reference pairs to use in each evaluation
            degradation_levels: Number of degradation levels to test
        
        Returns:
            Dictionary with test results, effectiveness metrics, and plot data
        """
        # Helper function to degrade text from original to completely random
        def degrade_text(text, degradation_level):
            # Apply exponential degradation to make the line decline faster
            # This makes even low degradation levels have a more significant effect
            # degradation_level^0.5 creates a curve that drops more quickly at first
            effective_degradation = degradation_level ** 0.25
            
            words = text.split()
            text_len = len(words)
            
            # Early return for empty text or zero degradation
            if text_len == 0 or degradation_level <= 0:
                return text
                
            # For 100% degradation, replace with completely random text
            if degradation_level >= 1.0:
                # Generate random words of random lengths
                random_words = []
                avg_word_len = sum(len(w) for w in words) / max(1, len(words))
                
                for _ in range(text_len):
                    word_len = max(1, int(random.gauss(avg_word_len, 2)))
                    random_word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') 
                                        for _ in range(word_len))
                    random_words.append(random_word)
                
                return ' '.join(random_words)
            
            # For partial degradation, replace a percentage of words
            num_to_replace = int(text_len * effective_degradation)
            
            # Also introduce word order changes for more dramatic degradation
            if effective_degradation > 0.3:
                # Shuffle a percentage of the text proportional to degradation level
                shuffle_percentage = (effective_degradation - 0.3) * 1.43  # scales from 0 to 1
                shuffle_positions = random.sample(range(text_len), int(text_len * shuffle_percentage))
                shuffle_words = [words[i] for i in shuffle_positions]
                random.shuffle(shuffle_words)
                for i, pos in enumerate(shuffle_positions):
                    words[pos] = shuffle_words[i]
                
            # Replace words with random ones
            positions = random.sample(range(text_len), min(num_to_replace, text_len))
            
            for pos in positions:
                # Generate a random word with similar length to the original
                orig_len = len(words[pos])
                word_len = max(1, int(random.gauss(orig_len, 2)))
                random_word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') 
                                     for _ in range(word_len))
                words[pos] = random_word
                
            return ' '.join(words)
        
        # Select random sentence pairs for testing
        valid_indices = [i for i in range(len(self.source_lines)) 
                        if len(self.source_lines[i].strip()) > 20 and len(self.target_lines[i].strip()) > 20]
        
        if len(valid_indices) < num_tests:
            raise ValueError(f"Not enough valid sentence pairs. Needed {num_tests}, found {len(valid_indices)}")
        
        test_indices = random.sample(valid_indices, num_tests)
        
        # Store results for plotting
        all_results = []
        
        # Add tqdm progress bar for the test indices
        for idx in tqdm(test_indices, desc="Testing sentence pairs", unit="pair"):
            source = self.source_lines[idx].strip()
            correct_translation = self.target_lines[idx].strip()
            
            # Evaluate correct translation
            baseline_score = self.evaluate_quality(source, correct_translation, example_size=num_pairs)
            
            # Create and evaluate degraded translations
            degraded_scores = []
            # Add tqdm progress bar for degradation levels
            for level in tqdm(range(1, degradation_levels + 1), 
                            desc=f"Testing degradation levels for pair {test_indices.index(idx)+1}/{num_tests}",
                            leave=False, unit="level"):
                # Convert level to percentage (0.0 to 1.0)
                degradation_percentage = level / degradation_levels
                degraded = degrade_text(correct_translation, degradation_percentage)
                score = self.evaluate_quality(source, degraded, example_size=num_pairs)
                degraded_scores.append(score)
            
            all_results.append({
                'source': source,
                'correct': correct_translation,
                'baseline_score': baseline_score,
                'degraded_scores': degraded_scores
            })
        
        # Calculate averages for plotting
        avg_baseline = np.mean([r['baseline_score'] for r in all_results])
        avg_degraded = [np.mean([r['degraded_scores'][i] for r in all_results]) 
                        for i in range(degradation_levels)]
        
        # Calculate effectiveness metrics
        # 1. Average drop from baseline to worst degradation
        avg_drop = avg_baseline - avg_degraded[-1]
        
        # 2. Percentage of tests where baseline > all degraded versions
        perfect_tests = sum(1 for r in all_results if r['baseline_score'] > max(r['degraded_scores']))
        perfect_test_pct = (perfect_tests / num_tests) * 100
        
        # 3. Monotonicity - does score consistently decrease as degradation increases?
        is_monotonic = all(avg_degraded[i] >= avg_degraded[i+1] for i in range(degradation_levels-1))
        
        # 4. Overall effectiveness score (0-100)
        effectiveness = min(100, max(0, int((avg_drop * 50) + (perfect_test_pct * 0.5))))
        
        # Create plot
        fig = plt.figure(figsize=(10, 6))
        
        # Plot individual test lines with low opacity
        for i, result in enumerate(all_results):
            plt.plot([0] + list(range(1, degradation_levels + 1)),
                    [result['baseline_score']] + result['degraded_scores'], 
                    'o-', alpha=0.3, label=f"Test {i+1}")
        
        # Plot average line with higher emphasis
        plt.plot([0] + list(range(1, degradation_levels + 1)), 
                [avg_baseline] + avg_degraded, 'bo-', linewidth=2, markersize=8, label="Average")
        
        # Add effectiveness information
        plt.text(0.02, 0.02, 
                 f"Effectiveness: {effectiveness}%\nPerfect tests: {perfect_test_pct:.1f}%\nAvg drop: {avg_drop:.2f}\nMonotonic: {'Yes' if is_monotonic else 'No'}", 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.xlabel('Degradation Level (0% to 100%)')
        plt.ylabel('Quality Score (Correlation)')
        plt.title('Translation Quality vs. Degradation Level')
        plt.grid(True)
        plt.xticks(range(0, degradation_levels + 1), 
                  ['0%'] + [f'{int(i/degradation_levels*100)}%' for i in range(1, degradation_levels + 1)])
        
        # Only show legend if there are few test cases
        if num_tests <= 5:
            plt.legend()
        else:
            plt.legend(['Average'] + [''] * num_tests)
        
        # Return results and figure
        return {
            'all_results': all_results,
            'avg_baseline': avg_baseline,
            'avg_degraded': avg_degraded,
            'effectiveness': effectiveness,
            'perfect_test_pct': perfect_test_pct,
            'avg_drop': avg_drop,
            'is_monotonic': is_monotonic,
            'figure': fig
        }

if __name__ == "__main__":
    corpus = PairCorpus(source_path="corpus/eng-engULB.txt", target_path="corpus/kos-kos.txt")
    # print(corpus.format_benchmark_examples(index_line=1000, top_n=5)[0])s
    # print(corpus.evaluate_quality(source_sentence="Hello, world!", draft_translation="Hello, world!", example_size=100))
    
    # Test with the single degradation method
    result = corpus.test_quality_metric(num_tests=10, num_pairs=75, degradation_levels=5)
    
    plt.show()