"""
Comprehensive validation pipeline for middle school writing evaluation.
"""
from typing import Dict, Any
from .metrics.scoring_logic import WritingScorer
from .metrics.writing_metrics import MiddleSchoolMetrics

class ValidationPipeline:
    def __init__(self):
        self.scorer = WritingScorer()
        self.metrics = MiddleSchoolMetrics()

    def validate_text(self, text: str) -> Dict[str, Any]:
        """Run comprehensive validation on text."""
        content_scores = self.scorer.score_content(text)
        writing_scores = self.metrics.get_comprehensive_score(text)

        # Combine scores with middle school specific criteria
        validation_results = {
            'content_analysis': content_scores,
            'writing_evaluation': writing_scores,
            'grade_level_assessment': {
                'readability': content_scores['readability'],
                'meets_standards': self._check_grade_standards(content_scores, writing_scores)
            }
        }

        return validation_results

    def _check_grade_standards(self, content: Dict[str, float],
                             writing: Dict[str, Any]) -> bool:
        """Check if text meets middle school standards."""
        # Grade level between 6-8
        grade_appropriate = 6 <= content['readability'] <= 8

        # Check writing criteria
        writing_criteria = all(
            score >= 0.7 for scores in writing.values()
            for score in scores.values()
        )

        return grade_appropriate and writing_criteria

    def generate_report(self, text: str) -> str:
        """Generate detailed validation report."""
        results = self.validate_text(text)

        report = ["=== Writing Validation Report ===\n"]
        report.append("Content Analysis:")
        for metric, score in results['content_analysis'].items():
            report.append(f"- {metric}: {score:.2f}")

        report.append("\nWriting Evaluation:")
        for category, scores in results['writing_evaluation'].items():
            report.append(f"\n{category}:")
            for aspect, score in scores.items():
                report.append(f"- {aspect}: {score:.2f}")

        report.append(f"\nGrade Level Assessment:")
        report.append(f"- Readability: Grade {results['grade_level_assessment']['readability']:.1f}")
        report.append(f"- Meets Standards: {'Yes' if results['grade_level_assessment']['meets_standards'] else 'No'}")

        return "\n".join(report)

if __name__ == "__main__":
    validator = ValidationPipeline()
    test_text = """
    The old bicycle stood in the corner of the garage, its once-shiny frame now
    covered in dust. Sarah remembered the summer days when she and her best friend
    would ride for hours, exploring every street in their neighborhood. Now, as she
    touched the handlebars, memories flooded back like warm sunshine.
    """
    print(validator.generate_report(test_text))
