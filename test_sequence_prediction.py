import json
import torch
from transformers import GPT2Tokenizer
from templates.nanoGPT_lite.sequence_testing import SequenceValidator

def prepare_sample(tokenizer, sample):
    """Prepare a single sample for evaluation"""
    # Tokenize input and target
    input_ids = tokenizer.encode(sample["prompt"], return_tensors="pt")[0]
    target_ids = tokenizer.encode(sample["target"], return_tensors="pt")[0]

    return {
        "input_ids": input_ids.tolist(),
        "target_ids": target_ids.tolist(),
        "prompt": sample["prompt"],
        "target": sample["target"]
    }

def run_test():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load test data
    with open("data/samples/test_data.json", "r") as f:
        raw_data = json.load(f)

    # Prepare tokenized samples
    test_data = [prepare_sample(tokenizer, sample) for sample in raw_data["samples"]]

    # Initialize validator
    validator = SequenceValidator()

    try:
        # Run text quality evaluation (without model)
        quality_results = []
        for sample in test_data:
            quality_score = validator.evaluate_completion_quality(sample["target"])
            quality_results.append({
                "prompt": sample["prompt"],
                "target": sample["target"],
                "quality_metrics": {
                    "overall_score": quality_score,
                    "vocabulary_level": validator._check_vocabulary_level(sample["target"]),
                    "coherence": validator._check_coherence(sample["target"]),
                    "sentence_structure": validator._check_sentence_structure(sample["target"])
                }
            })

        # Save results
        results = {
            "quality_evaluation": quality_results,
            "model_evaluation": "Model evaluation skipped - no model loaded"
        }

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Evaluation completed. Results saved to results.json")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    run_test()
