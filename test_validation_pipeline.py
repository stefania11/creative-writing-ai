from validation.validation_pipeline import ValidationPipeline

def main():
    validator = ValidationPipeline()

    # Test text appropriate for middle school
    test_text = """
    Sarah nervously approached the science fair table, her solar-powered robot
    gleaming under the fluorescent lights. "Will it work?" she wondered aloud.
    The judges watched intently as she pressed the start button. To everyone's
    amazement, the robot smoothly navigated the obstacle course, using only
    energy from its solar panels!
    """

    results = validator.validate_text(test_text)

    print("\nValidation Results:")
    print("==================")
    for category, metrics in results.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    main()
