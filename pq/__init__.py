"""
PQ (Pair Query) - A library for translation quality estimation, aligned translation pair retrieval, and benchmarking.

This package provides tools for:
- Managing parallel text corpora
- Measuring text similarity using various techniques
- Estimating translation quality
- Benchmarking quality metrics
- Machine translation with few-shot learning
"""

from pq.similarity_measures import (
    SimilarityMeasure,
    NGramSimilarity,
    WordOverlapSimilarity,
    LongestSubstringSimilarity,
    LengthSimilarity,
    WordEditDistanceSimilarity,
    TfidfCosineSimilarity,
    CombinedSimilarity
)

from pq.main import (
    SearchResult,
    PairCorpus,
    TextDegradation,
    QualityEstimator,
    Benchmarker
)

from pq.translator import (
    Translator,
    calculate_similarity,
    create_graphs,
    create_progressive_chart,
    run_benchmark
)

__all__ = [
    # Similarity measures
    'SimilarityMeasure',
    'NGramSimilarity',
    'WordOverlapSimilarity',
    'LongestSubstringSimilarity',
    'LengthSimilarity',
    'WordEditDistanceSimilarity',
    'TfidfCosineSimilarity',
    'CombinedSimilarity',
    
    # Main components
    'SearchResult',
    'PairCorpus',
    'TextDegradation',
    'QualityEstimator',
    'Benchmarker',
    
    # Translator components
    'Translator',
    'calculate_similarity',
    'create_graphs',
    'create_progressive_chart',
    'run_benchmark'
]
