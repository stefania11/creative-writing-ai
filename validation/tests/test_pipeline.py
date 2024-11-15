import unittest
from ..validation_pipeline import ValidationPipeline

class TestValidationPipeline(unittest.TestCase):
    def setUp(self):
        self.validator = ValidationPipeline()
        self.test_text = """
        The old bicycle stood in the corner of the garage, its once-shiny frame now
        covered in dust. Sarah remembered the summer days when she and her best friend
        would ride for hours, exploring every street in their neighborhood. Now, as she
        touched the handlebars, memories flooded back like warm sunshine.
        """

    def test_validation_results(self):
        results = self.validator.validate_text(self.test_text)
        self.assertIn('content_analysis', results)
        self.assertIn('writing_evaluation', results)
        self.assertIn('grade_level_assessment', results)

    def test_report_generation(self):
        report = self.validator.generate_report(self.test_text)
        self.assertIsInstance(report, str)
        self.assertIn('Writing Validation Report', report)
        self.assertIn('Content Analysis:', report)
        self.assertIn('Writing Evaluation:', report)
        self.assertIn('Grade Level Assessment:', report)

if __name__ == '__main__':
    unittest.main()
