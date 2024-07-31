# Language Translation Training Data Generator

This module provides tools for generating training data for language translation models, particularly focused on fine-tuning large language models like GPT-4. It consists of several components that work together to create high-quality, context-aware translation examples.

## Components

### 1. PairedData (pair.py)

The `PairedData` class is responsible for handling paired language data. It reads source and target language files, preprocesses the text, and provides methods for searching and retrieving training examples.

Key features:
- Reads and preprocesses source and target language files
- Supports custom preprocessing functions
- Implements a search functionality to find similar translations

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

### 4. GPT4oMiniAdapter (training_adapters.py)

This class is designed to generate training data specifically for fine-tuning GPT-4 or similar language models. It creates training examples in the format of conversation-like interactions, optionally including example translations for context.

Key features:
- Generates training data with system, user, and assistant messages
- Supports including example translations in the prompt
- Provides a method to save the generated training data to a file

## Usage

To use this module:

1. Prepare your source and target language files.
2. Choose or create appropriate preprocessing functions.
3. Initialize a `PairedData` object with your files and chosen query class.
4. Create a `GPT4oMiniAdapter` with the `PairedData` object.
5. Generate training data using the adapter's `generate_training_data` method.
6. Save the generated data using the `save_training_data` method.

This module provides a flexible and powerful way to generate high-quality training data for fine-tuning language translation models, with a focus on leveraging similar translations and context for improved results.
