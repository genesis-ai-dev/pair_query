import json
from typing import List, Dict
from pair import PairedData
from queries import Query
from datasets import Dataset
from huggingface_hub import HfApi


class GPT4oMiniAdapter:
    def __init__(self, paired_data: PairedData):
        self.paired_data = paired_data

    def generate_training_data(self, num_examples: int, include_examples: bool = False, k=10, start=0) -> List[Dict]:
        training_data = []
        for i in range(num_examples):
            i = i + start
            example = self.paired_data.get_training_example(i, k=k)
            messages = [
                {"role": "system", "content": "You are a translation assistant. Translate the given text accurately."}
            ]
            
            user_content = f"Translate this text: {example['query']}"
            
            if include_examples:
                examples_content = "Here are some example translations:\n"
                for idx, (source, target) in enumerate(example['results'][:3], 1):  # Use top 3 results
                    examples_content += f"-- {source} -> {target}\n"
                examples_content += f"\nUse the above examples to translate this:\n{example['query']}"
                user_content = examples_content
            
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": example['target']})
            
            training_data.append({"messages": messages})
        return training_data

    def save_training_data(self, filename: str, training_data: List[Dict]):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in training_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

class UnslothAdapter:
    def __init__(self, paired_data: PairedData):
        self.paired_data = paired_data

    def generate_training_data(self, num_examples: int, include_examples: bool = False) -> Dataset:
        instructions = []
        inputs = []
        outputs = []

        for i in range(num_examples):
            example = self.paired_data.get_training_example(i)
            
            instruction = "Translate the following text accurately."
            
            if include_examples:
                input_content = "Here are some example translations:\n"
                for idx, (source, target) in enumerate(example['results'][:3], 1):
                    input_content += f"-- {source} -> {target}\n"
                input_content += f"\nUse the above examples to translate this:\n{example['query']}"
            else:
                input_content = example['query']
            
            instructions.append(instruction)
            inputs.append(input_content)
            outputs.append(example['target'])

        dataset_dict = {
            "instruction": instructions,
            "input": inputs,
            "output": outputs
        }

        return Dataset.from_dict(dataset_dict)

    def save_training_data(self, dataset: Dataset, filename: str):
        dataset.save_to_disk(filename)

    def upload_to_huggingface(self, dataset: Dataset, repo_name: str, token: str):
        api = HfApi()
        api.create_repo(repo_name, private=True, token=token)
        dataset.push_to_hub(repo_name, token=token)
