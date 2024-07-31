import json
from typing import List, Dict
from pair import PairedData
from queries import Query

class GPT4oMiniAdapter:
    def __init__(self, paired_data: PairedData):
        self.paired_data = paired_data

    def generate_training_data(self, num_examples: int, include_examples: bool = False) -> List[Dict]:
        training_data = []
        for i in range(num_examples):
            example = self.paired_data.get_training_example(i)
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

