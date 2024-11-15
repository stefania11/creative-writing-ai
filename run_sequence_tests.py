import logging
logging.basicConfig(level=logging.INFO)
import os
import sys
import torch
import json
from templates.nanoGPT_lite.train_model import train_model
from templates.nanoGPT_lite.sequence_testing import SequenceValidator
from transformers import GPT2Tokenizer

def main():
    print("Starting model training...")
    checkpoint_path = train_model()

    print("\nInitializing sequence testing...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create test prompts suitable for middle school creative writing
    test_prompts = [
        # Story Development
        "Write a story about a student who discovers a hidden talent during a school talent show.",
        "Create a mystery story about a missing library book that leads to an unexpected adventure.",
        "Write about a day when all electronic devices stopped working at school.",

        # Descriptive Writing
        "Describe your perfect treehouse, including all the special features it would have.",
        "Write about a rainy day that turns into something magical.",
        "Describe a busy school cafeteria using all five senses.",

        # Character Development
        "Write about a shy student who becomes the hero of the day.",
        "Create a story about two friends who have very different personalities.",
        "Describe a day in the life of your pet from their perspective.",

        # Creative Prompts
        "What if you could trade places with your teacher for a day?",
        "Imagine you found a mysterious package in your locker...",
        "Write about a field trip that goes hilariously wrong."
    ]

    # Create test data with specific evaluation criteria
    test_data = []
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, truncation=True, max_length=512)
        # Add evaluation criteria specific to each prompt type
        test_data.append({
            'input_ids': input_ids,
            'target_ids': input_ids,
            'criteria': {
                'story_elements': ['plot', 'character', 'setting', 'conflict', 'resolution'],
                'writing_mechanics': ['grammar', 'punctuation', 'spelling'],
                'creativity': ['originality', 'imagination', 'engagement'],
                'grade_level': 'middle_school',
                'prompt_type': 'creative_writing'
            }
        })

    # Save test data
    os.makedirs('data', exist_ok=True)
    with open('data/test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print("\nRunning sequence prediction tests...")
    validator = SequenceValidator(checkpoint_path)
    results = validator.evaluate_sequence_prediction(test_data)

    # Save detailed results
    with open('data/sequence_evaluation_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTest Results:")
    print(f"Average Perplexity: {results.get('avg_perplexity', 0):.2f}")
    print(f"Average BLEU Score: {results.get('avg_bleu', 0):.2f}")
    print(f"Average Quality Score: {results.get('avg_quality', 0):.2f}")
    print("\nDetailed results saved to data/sequence_evaluation_report.json")

if __name__ == "__main__":
    main()
