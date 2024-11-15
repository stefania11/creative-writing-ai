"""
Validation pipeline for creative writing assessment.
"""
import spacy
import textstat
from typing import Dict, Any

class ValidationPipeline:
    def __init__(self):
        """Initialize the validation pipeline with spaCy model."""
        self.nlp = spacy.load('en_core_web_sm')

    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate text against middle school writing standards.
        """
        doc = self.nlp(text)

        # Calculate basic metrics
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))

        # Grade level assessment
        grade_level = {
            'readability': textstat.flesch_kincaid_grade(text),
            'complexity': textstat.gunning_fog(text),
            'reading_ease': textstat.flesch_reading_ease(text)
        }

        # Content analysis
        content_analysis = self._analyze_content(doc, word_count, sentence_count)

        # Writing mechanics
        writing_mechanics = self._analyze_mechanics(doc)

        return {
            'grade_level_assessment': grade_level,
            'content_analysis': content_analysis,
            'writing_mechanics': writing_mechanics,
            'statistics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0
            }
        }

    def _analyze_content(self, doc, word_count: int, sentence_count: int) -> Dict[str, float]:
        """Analyze content quality metrics."""
        # Vocabulary diversity
        unique_words = len({token.text.lower() for token in doc if not token.is_punct})
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0

        # Sentence complexity
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        sentence_complexity = min(1.0, avg_sentence_length / 20)  # Normalize to 0-1

        # Coherence (based on proper sentence structure)
        proper_sentences = sum(1 for sent in doc.sents if self._is_proper_sentence(sent))
        coherence = proper_sentences / sentence_count if sentence_count > 0 else 0

        return {
            'vocabulary_diversity': vocabulary_diversity,
            'sentence_complexity': sentence_complexity,
            'coherence': coherence
        }

    def _analyze_mechanics(self, doc) -> Dict[str, float]:
        """Analyze writing mechanics."""
        total_tokens = len(doc)
        if total_tokens == 0:
            return {'grammar': 0.0, 'punctuation': 0.0, 'spelling': 0.0}

        # Grammar check (simplified)
        grammar_errors = sum(1 for token in doc if token.dep_ == 'ROOT' and token.pos_ not in ['VERB', 'AUX'])
        grammar_score = 1.0 - (grammar_errors / total_tokens)

        # Punctuation check
        punct_tokens = [token for token in doc if token.is_punct]
        expected_punct = len(list(doc.sents))  # At least one punctuation per sentence
        punct_score = min(1.0, len(punct_tokens) / expected_punct if expected_punct > 0 else 0)

        # Spelling check (using spaCy's token.is_oov)
        misspelled = sum(1 for token in doc if not token.is_punct and token.is_oov)
        spelling_score = 1.0 - (misspelled / total_tokens)

        return {
            'grammar': max(0.0, min(1.0, grammar_score)),
            'punctuation': max(0.0, min(1.0, punct_score)),
            'spelling': max(0.0, min(1.0, spelling_score))
        }

    def _is_proper_sentence(self, sent) -> bool:
        """Check if a sentence has proper structure."""
        # Basic check for subject-verb structure
        has_subject = False
        has_verb = False

        for token in sent:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                has_subject = True
            if token.pos_ == 'VERB':
                has_verb = True

        return has_subject and has_verb
