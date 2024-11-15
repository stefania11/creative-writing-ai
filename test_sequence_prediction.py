import json
from templates.nanoGPT_lite.sequence_testing import SequenceValidator

def run_test():
    # Load test data
    with open("data/samples/test_data.json", "r") as f:
        test_data = json.load(f)

    # Initialize validator
    validator = SequenceValidator()

    # Run evaluation
    results = validator.evaluate_sequence_prediction(test_data["samples"])

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_test()
