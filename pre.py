import re
from typing import Callable

def remove_punctuation(text: str) -> str:
    """Remove all punctuation from the given text."""
    return re.sub(r'[^\w\s]', '', text).lower()

def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace, including leading and trailing spaces."""
    return ' '.join(text.split())

def remove_numbers(text: str) -> str:
    """Remove all numbers from the text."""
    return re.sub(r'\d+', '', text)

def compose_functions(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple string processing functions into a single function."""
    def composed_func(text: str) -> str:
        for func in funcs:
            text = func(text)
        return text
    return composed_func

# Example usage:
# pre_process = compose_functions(remove_punctuation, lowercase, remove_extra_whitespace)
# This creates a function that removes punctuation, converts to lowercase, and removes extra whitespace
