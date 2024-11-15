"""
Core scoring logic for middle school writing evaluation using NLP.
"""
import nltk
import spacy
import textstat
from typing import Dict, Any

class WritingScorer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        self.nlp = spacy.load('en_core_web_sm')

    def score_content(self, text: str) -> Dict[str, float]:
        """Score content based on coherence and development."""
        doc = self.nlp(text)
        return {
            'coherence': self._calculate_coherence(doc),
            'theme_development': self._analyze_theme(doc),
            'readability': textstat.flesch_kincaid_grade(text)
        }

    def _calculate_coherence(self, doc: Any) -> float:
        """Calculate text coherence score."""
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
        # Basic coherence based on sentence connections
        return min(len(sentences) / max(len(doc), 1), 1.0)

    def _analyze_theme(self, doc: Any) -> float:
        """Analyze theme development."""
        key_concepts = [token.text.lower() for token in doc
                       if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        return len(set(key_concepts)) / max(len(key_concepts), 1)

if __name__ == "__main__":
    scorer = WritingScorer()
    test_text = "The summer breeze rustled through the trees. Birds chirped their morning songs."
    print("Content scores:", scorer.score_content(test_text))
