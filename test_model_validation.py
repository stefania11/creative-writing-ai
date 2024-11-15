from validation.experiment.model_validator import ModelValidator

def main():
    validator = ModelValidator('path/to/model')
    test_cases = [
        {
            'prompt': 'Write a story about discovering an unexpected talent.',
            'expected_metrics': {
                'grade_level': {'min': 6, 'max': 8},
                'content_quality': 0.7
            }
        }
    ]
    results = validator.validate_model(test_cases)
    print("\nValidation Results:", results)

if __name__ == "__main__":
    main()
