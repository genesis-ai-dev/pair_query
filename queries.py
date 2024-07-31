from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz
import gensim.downloader as api

class Query:
    def __init__(self, source_lines: List[str], target_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.target_lines = target_lines
    
    def search(self, query: str, k: int):
        # search source_lines for query
        # return relevant lines and their corresponding target lines in list of tuples [(source_line, target_line)]
        # The source and target lines are aligned translation pairs, so each index corresponds to the same meaning in both
        pass

class TFiDF(Query):
    def __init__(self, source_lines: List[str], target_lines: List[str]) -> None:
        super().__init__(source_lines=source_lines, target_lines=target_lines)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.source_lines)
    
    def search(self, query: str, k: int):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(cosine_similarities)[-k:][::-1]
        
        return [(self.source_lines[idx], self.target_lines[idx]) for idx in top_k_indices]

class MostOverlappingWords(Query):
    def __init__(self, source_lines: List[str], target_lines: List[str]) -> None:
        super().__init__(source_lines=source_lines, target_lines=target_lines)
        self.source_words = [set(line.lower().split()) for line in self.source_lines]
    
    def search(self, query: str, k: int):
        query_words = set(query.lower().split())
        
        overlap_scores = []
        for idx, source_set in enumerate(self.source_words):
            overlap = len(query_words.intersection(source_set))
            overlap_scores.append((overlap, idx))
        
        top_k = sorted(overlap_scores, reverse=True)[:k]
        
        return [(self.source_lines[idx], self.target_lines[idx]) for _, idx in top_k]

class BM25Search(Query):
    def __init__(self, source_lines: List[str], target_lines: List[str]) -> None:
        super().__init__(source_lines, target_lines)
        tokenized_corpus = [line.lower().split() for line in source_lines]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, k: int):
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        
        return [(self.source_lines[idx], self.target_lines[idx]) for idx in top_k_indices]

class FuzzySearch(Query):
    def search(self, query: str, k: int):
        fuzzy_scores = [(fuzz.ratio(query.lower(), line.lower()), idx) for idx, line in enumerate(self.source_lines)]
        top_k = sorted(fuzzy_scores, reverse=True)[:k]
        
        return [(self.source_lines[idx], self.target_lines[idx]) for _, idx in top_k]

class WordEmbeddingSearch(Query):
    def __init__(self, source_lines: List[str], target_lines: List[str]) -> None:
        super().__init__(source_lines, target_lines)
        self.model = api.load("glove-wiki-gigaword-100")
        self.line_vectors = [self.get_line_vector(line) for line in source_lines]
    
    def get_line_vector(self, line: str):
        words = line.lower().split()
        return np.mean([self.model[word] for word in words if word in self.model], axis=0)
    
    def search(self, query: str, k: int):
        query_vector = self.get_line_vector(query)
        similarities = [cosine_similarity([query_vector], [vec])[0][0] for vec in self.line_vectors]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [(self.source_lines[idx], self.target_lines[idx]) for idx in top_k_indices]