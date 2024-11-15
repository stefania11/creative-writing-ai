"""
Validation pipeline for assessing creative writing against middle school standards.
"""
import spacy
import textstat
from typing import Dict, Any

class ValidationPipeline:
    def __init__(self):
        """Initialize the validation pipeline with required models."""
        self.nlp = spacy.load("en_core_web_sm")

    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate text against middle school writing standards.
        Returns comprehensive metrics about the text quality.
        """
        doc = self.nlp(text)

        return {
            'grade_level_assessment': self._assess_grade_level(text),
            'content_analysis': self._analyze_content(doc),
            'writing_mechanics': self._check_mechanics(doc),
            'creativity_metrics': self._assess_creativity(doc)
        }

    def _assess_grade_level(self, text: str) -> Dict[str, float]:
        """Assess the grade level of the text."""
        return {
            'readability': textstat.flesch_kincaid_grade(text),
            'reading_ease': textstat.flesch_reading_ease(text),
            'grade_level': textstat.coleman_liau_index(text)
        }

    def _analyze_content(self, doc) -> Dict[str, float]:
        """Analyze content quality and structure."""
        # Calculate basic metrics
        sentences = len(list(doc.sents))
        words = len([token for token in doc if not token.is_punct])

        # Analyze sentence structure
        sentence_lengths = [len([token for token in sent if not token.is_punct])
                          for sent in doc.sents]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        return {
            'coherence': min(1.0, sentences / 20),  # Normalize to 0-1
            'vocabulary_diversity': len(set([token.text.lower() for token in doc
                                          if not token.is_punct])) / words if words > 0 else 0,
            'sentence_complexity': avg_sentence_length / 15  # Normalize to typical middle school length
        }

    def _check_mechanics(self, doc) -> Dict[str, float]:
        """Check writing mechanics (grammar, punctuation, etc.)."""
        sentences = list(doc.sents)
        total_sentences = len(sentences)

        # Check for basic punctuation and capitalization
        proper_endings = sum(1 for sent in sentences
                           if sent.text.strip()[-1] in {'.', '!', '?'})
        capitalized = sum(1 for sent in sentences
                         if sent.text.strip()[0].isupper())

        return {
            'punctuation': proper_endings / total_sentences if total_sentences > 0 else 0,
            'capitalization': capitalized / total_sentences if total_sentences > 0 else 0,
            'grammar_score': 0.8  # Placeholder - would need more sophisticated grammar checking
        }

    def _assess_creativity(self, doc) -> Dict[str, float]:
        """Assess creative elements of the writing."""
        # Count unique descriptive words (adjectives and adverbs)
        descriptive_words = set([token.text.lower() for token in doc
                               if token.pos_ in {'ADJ', 'ADV'}])

        # Count dialogue instances
        dialogue_count = sum(1 for token in doc if token.text in {'"', '"', '"'}) / 2

        return {
            'descriptive_richness': len(descriptive_words) / len(doc) if len(doc) > 0 else 0,
            'dialogue_usage': min(1.0, dialogue_count / 5),  # Normalize to 0-1
            'vocabulary_sophistication': 0.7  # Placeholder - would need more sophisticated analysis
        }

