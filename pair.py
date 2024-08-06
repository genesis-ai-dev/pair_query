"""
Highlevel class for handling paired language data.
"""
from queries import Query
from typing import Callable, Type
import re

def rm(text):
    text = text.split(" ")
    "1co 3030 words"
    text = text[2:-1]
    return " ".join(text)
class PairedData:
    def __init__(self, source_file: str, target_file: str, pre_process: Callable[[str], str], query_class: Type[Query]) -> None:
        source_target_pairs = [
            (source_line, rm(target_line)) 
            for source_line, target_line in zip(self.read_file(source_file, pre_process), self.read_file(target_file, pre_process)) 
            if len(target_line) > 11
        ]
        print(len(source_target_pairs))
        self.source_lines, self.target_lines = zip(*source_target_pairs) if source_target_pairs else ([], [])
        self.query = query_class(self.source_lines, self.target_lines)
    
    def read_file(self, filename: str, pre_process: Callable[[str], str]) -> list[str]:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [pre_process(line.strip()) for line in lines]
    
    def search(self, query: str, k: int):
        return self.query.search(query=query, k=k)

    def get_training_example(self, index, k=10):
        query = self.source_lines[index]
        target = self.target_lines[index]
        results = self.search(query, k=k)  # Get top 5 results
        
        # Remove the exact match from results if present
        results = [r for r in results if r[0] != query]
        
        return {
            'query': query,
            'target': target,
            'results': results
        }

