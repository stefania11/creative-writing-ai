"""
Experiment runner for validation testing.
"""
import json
import os
from typing import Dict, Any
from .validation_pipeline import ValidationPipeline

class ValidationExperiment:
    def __init__(self, config_path: str):
        self.validator = ValidationPipeline()
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def run_experiment(self, output_dir: str = 'results') -> Dict[str, Any]:
        """Run validation experiment with test cases."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for test_case in self.config['experiment']['test_cases']:
            # Simulate text generation (replace with actual model generation)
            generated_text = """
            Sarah discovered her unusual talent during art class. While other students
            struggled with regular paintings, she found she could create beautiful
            artwork using only dots. Her pointillism technique amazed her teachers
            and classmates, leading to an unexpected journey of self-discovery.
            """

            # Run validation
            validation_results = self.validator.validate_text(generated_text)

            # Save results
            case_results = {
                'prompt': test_case['prompt'],
                'validation_results': validation_results,
                'meets_criteria': self._check_criteria(validation_results)
            }
            results[test_case['prompt']] = case_results

        # Save experiment results
        output_path = os.path.join(output_dir, 'validation_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _check_criteria(self, results: Dict[str, Any]) -> bool:
        """Check if results meet validation criteria."""
        criteria = self.config['experiment']['validation_criteria']

        # Check content quality
        content_scores = results['content_analysis']
        content_meets = all(
            content_scores.get(metric, 0) >= threshold
            for metric, threshold in criteria['content_quality'].items()
        )

        # Check grade level
        grade_level = results['grade_level_assessment']['readability']
        grade_appropriate = 6 <= grade_level <= 8

        return content_meets and grade_appropriate

if __name__ == "__main__":
    experiment = ValidationExperiment('templates/nanoGPT_lite/ideas/validation_experiment.json')
    results = experiment.run_experiment()
    print(json.dumps(results, indent=2))
