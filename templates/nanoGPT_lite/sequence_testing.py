import torch
import torch.nn.functional as F
from transformer_model import CreativeWritingTransformer
import json
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
import nltk

class SequenceValidator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            self.load_model(model_path)

        # Download required NLTK data
        try:
            nltk.download('punkt')
        except:
            pass

    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CreativeWritingTransformer(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def evaluate_sequence_prediction(self, test_data, max_length=100):
        """Evaluate model's sequence prediction capabilities"""
        results = {
            'perplexity': [],
            'bleu_scores': [],
            'completion_quality': []
        }

        for sample in test_data:
            # Tokenize input and target
            input_seq = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.device)
            target_seq = torch.tensor(sample['target_ids']).unsqueeze(0).to(self.device)

            # Generate completion
            with torch.no_grad():
                output = self.model.generate(input_seq, max_length=max_length)

                # Calculate perplexity
                logits = self.model(input_seq)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                perplexity = torch.exp(loss).item()

                # Calculate BLEU score
                generated_text = self.decode_tokens(output[0].cpu().numpy())
                target_text = self.decode_tokens(target_seq[0].cpu().numpy())
                bleu_score = sentence_bleu([target_text.split()], generated_text.split())

                # Store results
                results['perplexity'].append(perplexity)
                results['bleu_scores'].append(bleu_score)

                # Evaluate completion quality
                quality_score = self.evaluate_completion_quality(generated_text)
                results['completion_quality'].append(quality_score)

        return {
            'avg_perplexity': np.mean(results['perplexity']),
            'avg_bleu': np.mean(results['bleu_scores']),
            'avg_quality': np.mean(results['completion_quality'])
        }

    def evaluate_completion_quality(self, text):
        """Evaluate the quality of generated text"""
        # Implement quality metrics for middle school writing
        metrics = {
            'sentence_structure': self._check_sentence_structure(text),
            'vocabulary_level': self._check_vocabulary_level(text),
            'coherence': self._check_coherence(text)
        }
        return np.mean(list(metrics.values()))

    def _check_sentence_structure(self, text):
        """Check for proper sentence structure"""
        sentences = nltk.sent_tokenize(text)
        score = 0
        for sent in sentences:
            # Basic checks for capitalization and ending punctuation
            if sent[0].isupper() and sent[-1] in '.!?':
                score += 1
        return score / len(sentences) if sentences else 0

    def _check_vocabulary_level(self, text):
        """Check if vocabulary matches middle school level"""
        # Split text into words
        words = nltk.word_tokenize(text.lower())

        # Download required NLTK data if not already present
        try:
            nltk.download('words')
            nltk.download('averaged_perceptron_tagger')
        except:
            pass

        # Get English word list and create middle school vocabulary set
        word_list = set(nltk.corpus.words.words())

        # Count words that are in the word list
        valid_words = sum(1 for word in words if word in word_list)

        # Calculate basic statistics
        total_words = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0

        # Get parts of speech
        pos_tags = nltk.pos_tag(words)

        # Calculate complexity metrics
        complex_words = sum(1 for word in words if len(word) > 8)  # Words longer than 8 chars
        advanced_pos = sum(1 for _, tag in pos_tags if tag in ['JJR', 'JJS', 'RBR', 'RBS'])  # Comparative/superlative forms

        # Score based on multiple factors (1.0 is ideal for middle school)
        vocab_diversity = unique_words / total_words if total_words > 0 else 0
        word_complexity = complex_words / total_words if total_words > 0 else 0
        grammar_complexity = advanced_pos / total_words if total_words > 0 else 0

        # Combine scores with middle-school appropriate weights
        score = (
            0.4 * (0.7 if 0.3 <= vocab_diversity <= 0.6 else 0.3) +  # Vocabulary diversity
            0.3 * (0.8 if 4 <= avg_word_length <= 7 else 0.4) +     # Average word length
            0.3 * (0.9 if 0.05 <= word_complexity <= 0.15 else 0.5)  # Complex word ratio
        )

        return min(1.0, score)

    def _check_coherence(self, text):
        """Check text coherence and flow"""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0

        # Download required NLTK data if not already present
        try:
            nltk.download('stopwords')
        except:
            pass

        # Get stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))

        # Calculate sentence-level metrics
        scores = []

        for i in range(len(sentences) - 1):
            current = set(nltk.word_tokenize(sentences[i].lower())) - stopwords
            next_sent = set(nltk.word_tokenize(sentences[i + 1].lower())) - stopwords

            # Calculate overlap between consecutive sentences
            overlap = len(current.intersection(next_sent))
            total = len(current.union(next_sent))

            if total > 0:
                similarity = overlap / total
                scores.append(similarity)

        # Additional coherence metrics
        avg_sent_length = sum(len(nltk.word_tokenize(s)) for s in sentences) / len(sentences)
        sent_length_variance = np.var([len(nltk.word_tokenize(s)) for s in sentences])

        # Combine metrics into final coherence score
        base_coherence = np.mean(scores) if scores else 0.5
        length_score = 0.8 if 8 <= avg_sent_length <= 20 else 0.4  # Ideal middle school sentence length
        variance_score = 0.8 if sent_length_variance <= 25 else 0.4  # Some variance is good, but not too much

        final_score = (0.5 * base_coherence + 0.3 * length_score + 0.2 * variance_score)
        return min(1.0, final_score)

    def decode_tokens(self, token_ids):
        """Convert token IDs back to text"""
        if not hasattr(self, '_tokenizer'):
            try:
                # Try to get tokenizer from model
                self._tokenizer = self.model.get_tokenizer()
            except (AttributeError, TypeError):
                # Fallback to GPT2 tokenizer if model's tokenizer is not available
                from transformers import GPT2Tokenizer
                self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                print("Warning: Using fallback GPT2 tokenizer")

        # Convert numpy array to list if necessary
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        try:
            # Try to decode the tokens
            text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            print(f"Warning: Token decoding failed - {str(e)}")
            return "[DECODING_ERROR]"

if __name__ == "__main__":
    validator = SequenceValidator()
    # Add test execution code here
