from validation.experiment.model_integration import CreativeWritingModelValidator, TEST_PROMPTS

def main():
    print("Starting model validation...")
    validator = CreativeWritingModelValidator()
    results = validator.generate_and_validate(TEST_PROMPTS)

    # Print summary
    print("\nValidation Results Summary:")
    print("==========================")
    for prompt, result in results['test_cases'].items():
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Text Length: {len(result['generated_text'])} chars")
        print("Validation Metrics:")
        for metric_type, metrics in result['validation_results'].items():
            print(f"- {metric_type}: {metrics}")

if __name__ == "__main__":
    main()
