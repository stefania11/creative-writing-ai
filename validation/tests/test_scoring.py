import unittest
from ..metrics.scoring_logic import WritingScorer

class TestWritingScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = WritingScorer()
        self.test_text = "The summer breeze rustled through the trees. Birds chirped their morning songs."

    def test_content_scoring(self):
        scores = self.scorer.score_content(self.test_text)
        self.assertIn('coherence', scores)
        self.assertIn('theme_development', scores)
        self.assertIn('readability', scores)

if __name__ == '__main__':
    unittest.main()
