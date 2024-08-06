from pair import PairedData
from queries import Query, TFiDF, MostOverlappingWords, BM25Search
from pre import remove_punctuation, remove_extra_whitespace, remove_numbers, compose_functions
from training_adapters import GPT4oMiniAdapter


# Define preprocessing function
preprocess = compose_functions(remove_punctuation, remove_extra_whitespace)

# Initialize PairedData with source and target files
paired_data = PairedData(
    source_file="eng.txt",
    target_file="output.txt",
    pre_process=preprocess,
    query_class=BM25Search
)

# Create GPT4oMiniAdapter
adapter = GPT4oMiniAdapter(paired_data)
# Generate training data
num_examples = 12306 - 100
include_examples = True
training_data = adapter.generate_training_data(20, include_examples, k=12, start=12306-100)

# Save training data to a file
adapter.save_training_data("gpt4omini_val_data_large.jsonl", training_data)

print(f"Generated {num_examples} training examples and saved to gpt4omini_training_data.jsonl")