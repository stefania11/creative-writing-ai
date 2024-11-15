import unittest
from ..validation_framework import ValidationFramework

class TestValidationFramework(unittest.TestCase):
    def setUp(self):
        self.validator = ValidationFramework()
        self.test_text = "Test creative writing sample."

    def test_basic_validation(self):
        results = self.validator.validate_text(self.test_text)
        self.assertIsInstance(results, dict)
        self.assertIn('content', results)
        self.assertIn('structure', results)
        self.assertIn('language', results)

if __name__ == '__main__':
    unittest.main()
