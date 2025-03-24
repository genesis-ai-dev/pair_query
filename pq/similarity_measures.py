"""Module containing various text similarity measurement techniques."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np


class SimilarityMeasure(ABC):
    """Abstract base class for text similarity measurement techniques."""
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        pass


class NGramSimilarity(SimilarityMeasure):
    """N-gram based similarity measurement."""
    
    def __init__(self, max_n: int = 3):
        self.max_n = max_n
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def get_ngrams(text: str, n: int) -> Tuple[str, ...]:
        """Get n-grams from text."""
        words = text.lower().split()
        return tuple(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate n-gram similarity between two texts."""
        weights = {n: 2.0**n for n in range(1, self.max_n+1)}
        total_weight = sum(weights.values())
        
        weighted_sim = 0.0
        for n in range(1, self.max_n+1):
            if len(text1.split()) < n or len(text2.split()) < n:
                continue
                
            ngrams1 = set(self.get_ngrams(text1, n))
            ngrams2 = set(self.get_ngrams(text2, n))
            
            overlap = len(ngrams1 & ngrams2)
            union = len(ngrams1 | ngrams2)
            
            if union > 0:
                sim = overlap / union
                weighted_sim += sim * (weights[n] / total_weight)
        
        return weighted_sim


class WordOverlapSimilarity(SimilarityMeasure):
    """
    Similarity measure based on the proportion of shared words squared.
    
    This measure:
    1. Finds shared unique words between texts
    2. Divides by total unique words (Jaccard similarity)
    3. Squares the result to emphasize differences
    """
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate word overlap similarity between two texts."""
        # Get unique words from each text
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate intersection and union
        overlap = len(words1 & words2)  # shared words
        total = len(words1 | words2)    # all unique words
        
        # Calculate similarity as (shared words / total words)²
        if total == 0:
            return 0.0
            
        similarity = overlap / total
        return similarity ** 2  # Square to emphasize differences


class LongestSubstringSimilarity(SimilarityMeasure):
    """
    Similarity measure based on the longest common substring.
    
    This measure:
    1. Finds the longest common substring between two texts
    2. Normalizes by the length of the shorter text
    3. Squares the result to emphasize differences
    """
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
    
    def _find_longest_common_substring(self, text1: str, text2: str) -> str:
        """Find the longest common substring between two texts."""
        if not self.case_sensitive:
            text1 = text1.lower()
            text2 = text2.lower()
            
        # Create a table of size len(text1) x len(text2)
        m = len(text1)
        n = len(text2)
        
        # Initialize the table with zeros
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Variables to keep track of the maximum length and its ending position
        max_length = 0
        end_pos = 0
        
        # Fill the table
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
        
        # Extract the substring
        return text1[end_pos - max_length:end_pos]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on longest common substring."""
        if not text1 or not text2:
            return 0.0
            
        # Find the longest common substring
        lcs = self._find_longest_common_substring(text1, text2)
        
        # Calculate the similarity as (length of LCS / length of shorter text)²
        shorter_length = min(len(text1), len(text2))
        similarity = len(lcs) / shorter_length
        
        # Square to emphasize differences
        return similarity ** 2


class CombinedSimilarity(SimilarityMeasure):
    """Combines multiple similarity measures using joint probability or weighted average."""
    
    def __init__(self, 
                measures: List[SimilarityMeasure], 
                weights: Optional[List[float]] = None,
                mode: str = "multiply"):
        """
        Initialize with multiple similarity measures and optional weights.
        
        Args:
            measures: List of similarity measures to combine
            weights: Optional list of weights for each measure (must match length of measures)
                     If not provided, all measures are weighted equally
            mode: Combination mode - either "multiply" (joint probability) or "average" (weighted average)
        """
        if not measures:
            raise ValueError("At least one similarity measure must be provided")
            
        if weights and len(weights) != len(measures):
            raise ValueError("If weights are provided, they must match the number of measures")
            
        if mode not in ["multiply", "average"]:
            raise ValueError("Mode must be either 'multiply' or 'average'")
            
        self.measures = measures
        self.mode = mode
        
        # Normalize weights to sum to 1 if provided, otherwise use equal weights
        if weights:
            total = sum(weights)
            self.weights = [w / total for w in weights] if total > 0 else [1.0 / len(measures)] * len(measures)
        else:
            self.weights = [1.0 / len(measures)] * len(measures)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity by combining multiple measures.
        
        Uses the specified combination mode:
        - "multiply": Joint probability (Product(measure_i ^ weight_i))
        - "average": Weighted average (Sum(weight_i * measure_i))
        """
        # Calculate individual similarities
        similarities = [measure.calculate_similarity(text1, text2) for measure in self.measures]
        
        # Ensure all similarities are in [0,1] range
        similarities = [max(0.0, min(1.0, sim)) for sim in similarities]
        
        if self.mode == "multiply":
            # Joint probability (weighted product)
            result = 1.0
            for sim, weight in zip(similarities, self.weights):
                result *= sim ** weight
            return result
            
        else:  # mode == "average"
            # Weighted average
            return sum(sim * weight for sim, weight in zip(similarities, self.weights))


class LengthSimilarity(SimilarityMeasure):
    """
    Similarity measure based on the relative difference in text lengths.
    
    This measure:
    1. Compares the length of two texts
    2. Returns a ratio of shorter length / longer length
    3. Optionally squares the result to emphasize differences
    """
    
    def __init__(self, compare_words: bool = True, square_result: bool = True):
        """
        Initialize length similarity measure.
        
        Args:
            compare_words: If True, compares word count; otherwise character count
            square_result: If True, squares the similarity score to emphasize differences
        """
        self.compare_words = compare_words
        self.square_result = square_result
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on text length differences."""
        if not text1 or not text2:
            return 0.0
            
        # Get lengths based on words or characters
        if self.compare_words:
            len1 = len(text1.split())
            len2 = len(text2.split())
        else:
            len1 = len(text1)
            len2 = len(text2)
            
        # Handle empty texts
        if len1 == 0 and len2 == 0:
            return 1.0
        elif len1 == 0 or len2 == 0:
            return 0.0
            
        # Calculate ratio of shorter to longer length
        similarity = min(len1, len2) / max(len1, len2)
        
        # Optionally square the result
        if self.square_result:
            similarity = similarity ** 2
            
        return similarity 


class WordEditDistanceSimilarity(SimilarityMeasure):
    """
    Similarity measure based on word-level edit distance (Levenshtein distance).
    
    This measure:
    1. Splits texts into words
    2. Calculates edit distance between word sequences
    3. Normalizes by max length to get a value in [0,1]
    4. Optionally squares to emphasize differences
    """
    
    def __init__(self, case_sensitive: bool = False, square_result: bool = True):
        """
        Initialize word edit distance similarity measure.
        
        Args:
            case_sensitive: If False, words are compared case-insensitively
            square_result: If True, squares the similarity score to emphasize differences
        """
        self.case_sensitive = case_sensitive
        self.square_result = square_result
    
    def _levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate the Levenshtein distance between two sequences.
        
        Args:
            seq1: First sequence of words
            seq2: Second sequence of words
            
        Returns:
            The minimum number of single-element edits to transform seq1 into seq2
        """
        # Create a matrix of size (len(seq1)+1) x (len(seq2)+1)
        rows = len(seq1) + 1
        cols = len(seq2) + 1
        
        # Initialize the matrix
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Fill first row and column
        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j
        
        # Fill the rest of the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if seq1[i-1] == seq2[j-1]:
                    # No operation needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Minimum of deletion, insertion, or substitution
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        # The bottom-right cell contains the answer
        return dp[rows-1][cols-1]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on word-level edit distance."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Split texts into words
        if self.case_sensitive:
            words1 = text1.split()
            words2 = text2.split()
        else:
            words1 = text1.lower().split()
            words2 = text2.lower().split()
        
        # Calculate edit distance
        distance = self._levenshtein_distance(words1, words2)
        
        # Normalize by the maximum possible distance
        max_distance = max(len(words1), len(words2))
        if max_distance == 0:
            return 1.0
            
        # Convert to similarity (1 - normalized_distance)
        similarity = 1.0 - (distance / max_distance)
        
        # Optionally square the result
        if self.square_result:
            similarity = similarity ** 2
            
        return similarity 


class TfidfCosineSimilarity(SimilarityMeasure):
    """
    Similarity measure based on TF-IDF vectors and cosine similarity.
    
    This measure:
    1. Converts texts to TF-IDF vectors
    2. Calculates cosine similarity between vectors
    3. Returns a value in [0,1] where 1 indicates identical content
    """
    
    def __init__(self, min_n: int = 1, max_n: int = 2, lowercase: bool = True, 
                 analyzer: str = 'word', square_result: bool = False):
        """
        Initialize TF-IDF cosine similarity measure.
        
        Args:
            min_n: Minimum n-gram size
            max_n: Maximum n-gram size
            lowercase: Whether to convert text to lowercase
            analyzer: Feature type ('word' or 'char')
            square_result: If True, squares the similarity score to emphasize differences
        """
        self.min_n = min_n
        self.max_n = max_n
        self.lowercase = lowercase
        self.analyzer = analyzer
        self.square_result = square_result
        
        # We'll create the vectorizer on-demand for each calculation
        # because we need to fit it on the specific texts being compared
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF vectors and cosine similarity."""
        if not text1 or not text2:
            return 0.0
            
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=self.lowercase,
            ngram_range=(self.min_n, self.max_n),
            analyzer=self.analyzer
        )
        
        try:
            # Transform texts to TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Handle potential numerical issues
            similarity = max(0.0, min(1.0, cos_sim))
            
            # Optionally square the result
            if self.square_result:
                similarity = similarity ** 2
                
            return similarity
            
        except Exception as e:
            # Handle cases where vectorization fails (e.g., empty documents after preprocessing)
            return 0.0 