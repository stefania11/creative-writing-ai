import torch
from templates.nanoGPT_lite.transformer_model import CreativeWritingTransformer
from templates.nanoGPT_lite.sequence_testing import SequenceValidator
import json

def test_model_generation():
    """Test the model's text generation capabilities"""
    # Initialize model with smaller config for testing
    model_config = {
        'vocab_size': 50257,  # GPT-2 vocabulary size
        'n_embd': 256,       # Smaller embedding size for testing
        'n_head': 8,         # Number of attention heads
        'n_layer': 4,        # Number of transformer layers
        'dropout': 0.1
    }

    model = CreativeWritingTransformer(**model_config)
    validator = SequenceValidator()

    # Test prompts suitable for middle school writing
    test_prompts = [
        "Write a story about a student who discovers a hidden talent.",
        "Describe your perfect day at school.",
        "What would happen if animals could talk?",
    ]

    results = []

    for prompt in test_prompts:
        # Tokenize prompt
        tokenizer = model.get_tokenizer()
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate text
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50
                )

            # Decode generated text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Evaluate quality
            quality_score = validator.evaluate_completion_quality(generated_text)

            # Store results
            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'quality_score': quality_score,
                'metrics': {
                    'sentence_structure': validator._check_sentence_structure(generated_text),
                    'vocabulary_level': validator._check_vocabulary_level(generated_text),
                    'coherence': validator._check_coherence(generated_text)
                }
            }
            results.append(result)

            print(f"\nPrompt: {prompt}")
            print(f"Generated Text: {generated_text}")
            print(f"Quality Score: {quality_score}")
            print("Detailed Metrics:")
            for metric, score in result['metrics'].items():
                print(f"- {metric}: {score:.2f}")

        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {str(e)}")

    # Save results
    with open('generation_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def test_sequence_prediction():
    """Test the model's sequence prediction capabilities"""
    # Load test data
    try:
        with open('data/samples/test_data.json', 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Test data not found. Creating sample test data...")
        test_data = [
            {
                'input_ids': list(range(10)),  # Sample input tokens
                'target_ids': list(range(10, 20))  # Sample target tokens
            }
        ]
        with open('data/samples/test_data.json', 'w') as f:
            json.dump(test_data, f, indent=2)

    # Initialize model and validator
    model_config = {
        'vocab_size': 50257,
        'n_embd': 256,
        'n_head': 8,
        'n_layer': 4,
        'dropout': 0.1
    }

    model = CreativeWritingTransformer(**model_config)
    validator = SequenceValidator()

    # Test sequence prediction
    try:
        results = validator.evaluate_sequence_prediction(test_data)
        print("\nSequence Prediction Results:")
        print(f"Average Perplexity: {results['avg_perplexity']:.2f}")
        print(f"Average BLEU Score: {results['avg_bleu']:.2f}")
        print(f"Average Quality Score: {results['avg_quality']:.2f}")

        # Save results
        with open('sequence_prediction_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        print(f"Error in sequence prediction testing: {str(e)}")

if __name__ == "__main__":
    print("Testing Model Generation...")
    test_model_generation()

    print("\nTesting Sequence Prediction...")
    test_sequence_prediction()
