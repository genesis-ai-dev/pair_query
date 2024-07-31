"""
Highlevel class for handling paired language data.
"""
from queries import Query
from typing import Callable, Type

class PairedData:
    def __init__(self, source_file: str, target_file: str, pre_process: Callable[[str], str], query_class: Type[Query]) -> None:
        self.source_lines = self.read_file(source_file, pre_process)
        self.target_lines = self.read_file(target_file, pre_process)
        self.query = query_class(self.source_lines, self.target_lines)
    
    def read_file(self, filename: str, pre_process: Callable[[str], str]) -> list[str]:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [pre_process(line.strip()) for line in lines]
    
    def search(self, query: str, k: int):
        return self.query.search(query=query, k=k)

    def get_training_example(self, index):
        query = self.source_lines[index]
        target = self.target_lines[index]
        results = self.search(query, k=5)  # Get top 5 results
        
        # Remove the exact match from results if present
        results = [r for r in results if r[0] != query]
        
        return {
            'query': query,
            'target': target,
            'results': results
        }

