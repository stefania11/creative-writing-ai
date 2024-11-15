"""
Complete validation test for creative writing model.
"""
import os
import json
from datetime import datetime
from validation.experiment.model_integration import CreativeWritingModelValidator, TEST_PROMPTS

def run_validation_test():
    print("\n=== Starting Creative Writing Model Validation ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Test Prompts:")
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"{i}. {prompt}")
    print("\n=== Generating and Validating Text Samples ===")

    try:
        validator = CreativeWritingModelValidator()
        results = validator.generate_and_validate(TEST_PROMPTS)

        # Save detailed results
        results_file = 'results/validation_results.json'
        os.makedirs('results', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n=== Validation Results Summary ===")
        if 'test_cases' in results:
            for prompt, result in results['test_cases'].items():
                print(f"\nPrompt: {prompt[:50]}...")
                print(f"Generated Text Length: {len(result['generated_text'])} chars")
                print("Validation Metrics:")
                for metric_type, metrics in result['validation_results'].items():
                    print(f"- {metric_type}: {metrics}")

        if 'errors' in results and results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"- {error}")

        print(f"\nDetailed results saved to: {results_file}")

    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        raise

if __name__ == "__main__":
    run_validation_test()
