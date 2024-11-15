import sys
import json
from templates.nanoGPT_lite.sequence_testing import SequenceValidator
from templates.nanoGPT_lite.evaluation_metrics import evaluate_writing_quality

def run_validation_tests():
    # Test cases for middle school writing
    test_cases = [
        {
            "prompt": "Write a story about overcoming a challenge",
            "expected_elements": ["character development", "conflict", "resolution"]
        },
        {
            "prompt": "Describe your perfect day",
            "expected_elements": ["setting", "sensory details", "personal voice"]
        }
    ]

    try:
        validator = SequenceValidator()
        results = []

        for test_case in test_cases:
            # Generate text
            generated_text = validator.generate_text(test_case["prompt"])

            # Evaluate writing quality
            metrics = evaluate_writing_quality(generated_text)

            # Check for expected elements
            elements_present = all(
                element.lower() in generated_text.lower()
                for element in test_case["expected_elements"]
            )

            results.append({
                "prompt": test_case["prompt"],
                "generated_text": generated_text,
                "metrics": metrics,
                "elements_present": elements_present
            })

        # Save results
        with open("validation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Validation tests completed successfully")
        return True

    except Exception as e:
        print(f"Error during validation: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)
