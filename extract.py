from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        

if __name__ == "__main__":
    corpus = PairCorpus(source_path="corpus/eng-engULB.txt", target_path="corpus/kos-kos.txt")
    print(corpus.format_benchmark_examples(index_line=1000, top_n=5)[0])