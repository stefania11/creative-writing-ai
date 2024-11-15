import json
from validation.experiment.model_integration import CreativeWritingModelValidator, TEST_PROMPTS

def main():
    print("Starting model validation...")
    validator = CreativeWritingModelValidator()
    results = validator.generate_and_validate(TEST_PROMPTS)

    print("\nValidation Results Summary:")
    print("==========================")
    print(f"Total test cases: {results['overall_performance']['total_cases']}")
    print(f"Passing cases: {results['overall_performance']['passing_cases']}")
    print(f"Success rate: {results['overall_performance']['success_rate']:.2f}%")

    print("\nDetailed Results:")
    for prompt, case_results in results['test_cases'].items():
        print(f"\nPrompt: {prompt}")
        print(f"Meets criteria: {case_results['meets_criteria']}")
        if 'validation_results' in case_results:
            print("Validation metrics:")
            print(json.dumps(case_results['validation_results'], indent=2))

if __name__ == "__main__":
    main()
