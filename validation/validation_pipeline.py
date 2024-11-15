"""
Validation pipeline for middle school creative writing assessment.
"""
import spacy
import textstat
from typing import Dict, Any

class ValidationPipeline:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def validate_text(self, text: str) -> Dict[str, Any]:
        """Run comprehensive validation on text."""
        doc = self.nlp(text)
        return {
            'grade_level_assessment': self._assess_grade_level(text),
            'content_analysis': self._analyze_content(doc),
            'writing_evaluation': self._evaluate_writing(doc)
        }

    def _assess_grade_level(self, text: str) -> Dict[str, float]:
        """Assess grade level appropriateness."""
        return {
            'readability': textstat.flesch_kincaid_grade(text),
            'complexity': textstat.gunning_fog(text),
            'reading_ease': textstat.flesch_reading_ease(text)
        }

    def _analyze_content(self, doc) -> Dict[str, float]:
        """Analyze content quality and structure."""
        sentences = len(list(doc.sents))
        words = len([token for token in doc if not token.is_punct])
        return {
            'coherence': min(1.0, sentences / 20),
            'vocabulary_diversity': len(set([token.text.lower() for token in doc if not token.is_punct])) / words,
            'sentence_complexity': sum(len(list(sent)) for sent in doc.sents) / sentences
        }

    def _evaluate_writing(self, doc) -> Dict[str, Dict[str, float]]:
        """Evaluate writing mechanics and style."""
        return {
            'mechanics': self._check_mechanics(doc),
            'style': self._analyze_style(doc)
        }

    def _check_mechanics(self, doc) -> Dict[str, float]:
        total_sentences = len(list(doc.sents))
        return {
            'grammar': 1.0,  # Placeholder for now
            'punctuation': sum(1 for token in doc if token.is_punct) / total_sentences,
            'capitalization': sum(1 for token in doc if token.text[0].isupper()) / total_sentences
        }

    def _analyze_style(self, doc) -> Dict[str, float]:
        total_words = len([token for token in doc if not token.is_punct])
        return {
            'vocabulary_level': sum(len(token.text) for token in doc if not token.is_punct) / total_words,
            'sentence_variety': len(set(sent.root.pos_ for sent in doc.sents)) / len(list(doc.sents)),
            'descriptive_richness': len([token for token in doc if token.pos_ in ['ADJ', 'ADV']]) / total_words
        }
