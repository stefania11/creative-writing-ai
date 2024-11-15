"""
Model validator for creative writing assessment.
"""
from typing import Dict, Any, List
from ..validation_pipeline import ValidationPipeline

class ModelValidator:
    def __init__(self, model_path: str):
        """Initialize the model validator."""
        self.model_path = model_path
        self.validation_pipeline = ValidationPipeline()

    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate a single piece of text against middle school standards.
        """
        return self.validation_pipeline.validate_text(text)

    def validate_model(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate model performance across multiple test cases.
        """
        results = {
            'overall_metrics': {},
            'individual_results': []
        }

        for test_case in test_cases:
            validation_result = self.validate_text(test_case['generated_text'])
            results['individual_results'].append({
                'prompt': test_case['prompt'],
                'metrics': validation_result,
                'expected_metrics': test_case['expected_metrics']
            })

        # Calculate overall metrics
        total_cases = len(test_cases)
        passing_cases = sum(1 for result in results['individual_results']
                          if self._meets_standards(result['metrics']))

        results['overall_metrics'] = {
            'total_cases': total_cases,
            'passing_cases': passing_cases,
            'success_rate': (passing_cases / total_cases * 100) if total_cases > 0 else 0
        }

        return results

    def _meets_standards(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics meet middle school standards.
        """
        grade_level = metrics['grade_level_assessment']['readability']
        if not (6 <= grade_level <= 8):
            return False

        # Check content quality
        content_scores = metrics['content_analysis']
        if not all(score >= 0.7 for score in content_scores.values()):
            return False

        # Check writing mechanics
        mechanics_scores = metrics['writing_mechanics']
        if not all(score >= 0.7 for score in mechanics_scores.values()):
            return False

        return True
