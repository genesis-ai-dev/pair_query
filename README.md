# Language Translation Training Data Generator

This module provides tools for generating training data for language translation models, particularly focused on fine-tuning large language models like GPT-4. It consists of several components that work together to create high-quality, context-aware translation examples.

## Components

### 1. PairedData (pair.py)

The `PairedData` class is responsible for handling paired language data. It reads source and target language files, preprocesses the text, and provides methods for searching and retrieving training examples.

Key features:
- Reads and preprocesses source and target language files
- Supports custom preprocessing functions
- Implements a search functionality to find similar translations
- Provides a method to get training examples

### 2. Query Classes (queries.py)

This file contains various query classes that implement different search algorithms for finding similar translations. These include:

- TFiDF: Uses Term Frequency-Inverse Document Frequency for similarity matching
- MostOverlappingWords: Finds translations with the most overlapping words
- BM25Search: Implements the BM25 ranking function for information retrieval
- FuzzySearch: Uses fuzzy string matching for finding similar translations
- WordEmbeddingSearch: Utilizes word embeddings for semantic similarity search

### 3. Preprocessing Functions (pre.py)

The `pre.py` file provides utility functions for text preprocessing, including:

- Removing punctuation
- Removing extra whitespace
- Removing numbers
- Composing multiple preprocessing functions

### 4. Training Adapters (training_adapters.py)

This file contains classes for generating training data for different model architectures:

#### GPT4oMiniAdapter

Designed to generate training data specifically for fine-tuning GPT-4 or similar language models. It creates training examples in the format of conversation-like interactions, optionally including example translations for context.

Key features:
- Generates training data with system, user, and assistant messages
- Supports including example translations in the prompt
- Provides a method to save the generated training data to a file

#### UnslothAdapter

Generates training data in a format suitable for the Unsloth library, which is designed for efficient fine-tuning of language models.

Key features:
- Creates a dataset with instructions, inputs, and outputs
- Supports including example translations in the input
- Provides methods to save the dataset locally or upload it to Hugging Face

## Usage

To use this module:

1. Prepare your source and target language files.
2. Choose or create appropriate preprocessing functions.
3. Initialize a `PairedData` object with your files and chosen query class.
4. Create a `GPT4oMiniAdapter` or `UnslothAdapter` with the `PairedData` object.
5. Generate training data using the adapter's `generate_training_data` method.
6. Save the generated data using the appropriate save method.

## Example
```python

# Define preprocessing function
preprocess = compose_functions(remove_punctuation, remove_extra_whitespace)

# Initialize PairedData with source and target files
paired_data = PairedData(
    source_file="pair_query/tiny_corpus/source.txt",
    target_file="pair_query/tiny_corpus/target.txt",
    pre_process=preprocess,
    query_class=BM25Search
)

# Create GPT4oMiniAdapter
adapter = GPT4oMiniAdapter(paired_data)

# Generate training data
num_examples = 5
include_examples = True
training_data = adapter.generate_training_data(num_examples, include_examples)

# Save training data to a file
adapter.save_training_data("gpt4omini_training_data.jsonl", training_data)

print(f"Generated {num_examples} training examples and saved to gpt4omini_training_data.jsonl")
```

NOTE: You can also use the Unsloth adapter to generate a Hugginface instruction dataset for Llama 3.1 finetuning.