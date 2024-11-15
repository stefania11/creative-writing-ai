from validation.validation_pipeline import ValidationPipeline

def test_validation():
    validator = ValidationPipeline()

    # Test text appropriate for middle school
    test_text = """
    The old bicycle sat in the corner of Sarah's garage, covered in dust and cobwebs.
    She had forgotten about it until today, when her mom suggested cleaning out the garage.
    As Sarah wiped away years of neglect, she discovered something extraordinary:
    a small compartment hidden in the frame, containing a mysterious map.
    Her heart raced with excitement as she unfolded the weathered paper.
    """

    print("Running validation test...")
    results = validator.validate_text(test_text)

    print("\nValidation Results:")
    print("==================")
    print(f"Grade Level Assessment: {results['grade_level_assessment']}")
    print(f"Content Analysis: {results['content_analysis']}")
    print(f"Writing Evaluation: {results['writing_evaluation']}")

if __name__ == "__main__":
    test_validation()
