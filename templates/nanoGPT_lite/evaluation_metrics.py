import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from typing import Dict, List, Any

def _evaluate_story_elements(text: str) -> Dict[str, float]:
    """Evaluate basic story elements (plot, character, setting, etc.)"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Initialize scores
    scores = {
        'plot': 0.0,
        'character': 0.0,
        'setting': 0.0,
        'conflict': 0.0,
        'resolution': 0.0
    }

    # Character evaluation
    character_indicators = ['he', 'she', 'they', 'name', 'person', 'student', 'teacher', 'friend']
    character_mentions = sum(1 for word in words if word in character_indicators)
    scores['character'] = min(1.0, character_mentions / 5)

    # Setting evaluation
    setting_indicators = ['at', 'in', 'on', 'where', 'place', 'room', 'school', 'house']
    setting_mentions = sum(1 for word in words if word in setting_indicators)
    scores['setting'] = min(1.0, setting_mentions / 3)

    # Plot and conflict evaluation
    plot_indicators = ['then', 'after', 'before', 'when', 'finally', 'suddenly']
    conflict_indicators = ['but', 'however', 'though', 'problem', 'tried', 'couldn\'t']

    plot_mentions = sum(1 for word in words if word in plot_indicators)
    conflict_mentions = sum(1 for word in words if word in conflict_indicators)

    scores['plot'] = min(1.0, plot_mentions / 4)
    scores['conflict'] = min(1.0, conflict_mentions / 3)

    # Resolution evaluation (check if last few sentences contain resolution indicators)
    resolution_indicators = ['finally', 'resolved', 'solved', 'ended', 'learned', 'realized']
    last_sentences = ' '.join(sentences[-2:]).lower()
    resolution_score = sum(1 for word in resolution_indicators if word in last_sentences)
    scores['resolution'] = min(1.0, resolution_score / 2)

    return scores

def _evaluate_writing_mechanics(text: str) -> Dict[str, float]:
    """Evaluate writing mechanics (grammar, punctuation, spelling)"""
    sentences = sent_tokenize(text)

    # Initialize scores
    scores = {
        'grammar': 0.0,
        'punctuation': 0.0,
        'spelling': 0.0
    }

    # Grammar evaluation (basic checks)
    grammar_score = 0
    for sent in sentences:
        words = word_tokenize(sent)
        if words:
            # Check for capitalization
            if words[0][0].isupper():
                grammar_score += 1
            # Check for subject-verb basic structure
            if len(words) >= 3:
                grammar_score += 1
    scores['grammar'] = min(1.0, grammar_score / (len(sentences) * 2))

    # Punctuation evaluation
    punctuation_score = 0
    for sent in sentences:
        if sent[-1] in '.!?':
            punctuation_score += 1
        if ',' in sent:
            punctuation_score += 0.5
    scores['punctuation'] = min(1.0, punctuation_score / len(sentences))

    # Spelling evaluation (using NLTK words corpus)
    try:
        nltk.download('words', quiet=True)
        word_list = set(nltk.corpus.words.words())
        words = word_tokenize(text.lower())
        valid_words = sum(1 for word in words if word in word_list)
        scores['spelling'] = valid_words / len(words) if words else 0
    except:
        scores['spelling'] = 0.8  # Default if NLTK words unavailable

    return scores

def _evaluate_creativity(text: str, prompt: str) -> Dict[str, float]:
    """Evaluate creativity aspects of the writing"""
    scores = {
        'originality': 0.0,
        'imagination': 0.0,
        'engagement': 0.0
    }

    # Originality (compare with prompt)
    prompt_words = set(word_tokenize(prompt.lower()))
    text_words = set(word_tokenize(text.lower()))
    new_words = len(text_words - prompt_words)
    scores['originality'] = min(1.0, new_words / 20)

    # Imagination (check for descriptive language)
    descriptive_indicators = ['like', 'as', 'seemed', 'felt', 'looked', 'sounded', 'smelled']
    words = word_tokenize(text.lower())
    imagination_score = sum(1 for word in words if word in descriptive_indicators)
    scores['imagination'] = min(1.0, imagination_score / 5)

    # Engagement (sentence variety and dialog)
    sentences = sent_tokenize(text)
    sentence_types = {
        'question': sum(1 for s in sentences if s.endswith('?')),
        'exclamation': sum(1 for s in sentences if s.endswith('!')),
        'dialog': text.count('"') // 2
    }
    engagement_score = sum(sentence_types.values())
    scores['engagement'] = min(1.0, engagement_score / 5)

    return scores

def _check_grade_level(text: str) -> float:
    """Check if the text is appropriate for middle school level"""
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)

    if not words or not sentences:
        return 0.0

    # Calculate basic metrics
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = len(words) / len(sentences)

    # Score based on middle school appropriate ranges
    word_length_score = 1.0 if 4 <= avg_word_length <= 7 else 0.5
    sentence_length_score = 1.0 if 8 <= avg_sentence_length <= 20 else 0.5

    # Vocabulary complexity
    complex_words = sum(1 for word in words if len(word) > 8)
    complexity_ratio = complex_words / len(words)
    complexity_score = 1.0 if 0.05 <= complexity_ratio <= 0.15 else 0.5

    return (word_length_score + sentence_length_score + complexity_score) / 3

def _calculate_avg_metric(metrics: Dict[str, float]) -> float:
    """Calculate average of metrics"""
    return sum(metrics.values()) / len(metrics) if metrics else 0.0

def _summarize_writing_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize writing metrics across all samples"""
    summary = {
        'story_elements': {
            'plot': [], 'character': [], 'setting': [],
            'conflict': [], 'resolution': []
        },
        'writing_mechanics': {
            'grammar': [], 'punctuation': [], 'spelling': []
        },
        'creativity': {
            'originality': [], 'imagination': [], 'engagement': []
        },
        'grade_level_appropriateness': []
    }

    for metrics in metrics_list:
        for category in ['story_elements', 'writing_mechanics', 'creativity']:
            for key in summary[category]:
                summary[category][key].append(metrics[category][key])
        summary['grade_level_appropriateness'].append(
            metrics['grade_level_appropriateness']
        )

    # Calculate averages
    result = {}
    for category in summary:
        if isinstance(summary[category], dict):
            result[category] = {
                key: np.mean(values) for key, values in summary[category].items()
            }
        else:
            result[category] = np.mean(summary[category])

    return result
