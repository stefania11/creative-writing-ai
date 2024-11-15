import torch
import numpy as np
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
            test_samples = test_data['samples']
    except FileNotFoundError:
        print("Test data not found. Please ensure test_data.json exists in data/samples/")
        return
    except KeyError:
        print("Invalid test data format. Expected 'samples' key in test_data.json")
        return

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
        results = []
        for sample in test_samples:
            # Convert sample data to tensors
            input_tensor = torch.tensor(sample['input_ids']).unsqueeze(0)
            target_tensor = torch.tensor(sample['target_ids']).unsqueeze(0)

            # Get model predictions
            with torch.no_grad():
                logits = model(input_tensor)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tensor.view(-1)
                )

            # Evaluate quality of target text
            quality_score = validator.evaluate_completion_quality(sample['target'])

            result = {
                'prompt': sample['prompt'],
                'target': sample['target'],
                'loss': float(loss.item()),
                'quality_score': quality_score,
                'metrics': {
                    'sentence_structure': validator._check_sentence_structure(sample['target']),
                    'vocabulary_level': validator._check_vocabulary_level(sample['target']),
                    'coherence': validator._check_coherence(sample['target'])
                }
            }
            results.append(result)

            print(f"\nPrompt: {sample['prompt']}")
            print(f"Target: {sample['target']}")
            print(f"Loss: {result['loss']:.4f}")
            print(f"Quality Score: {quality_score:.2f}")
            print("Detailed Metrics:")
            for metric, score in result['metrics'].items():
                print(f"- {metric}: {score:.2f}")

        # Calculate averages
        avg_results = {
            'avg_loss': float(np.mean([r['loss'] for r in results])),
            'avg_quality': float(np.mean([r['quality_score'] for r in results])),
            'avg_metrics': {
                metric: float(np.mean([r['metrics'][metric] for r in results]))
                for metric in results[0]['metrics']
            }
        }

        # Save results
        with open('sequence_prediction_results.json', 'w') as f:
            json.dump({
                'individual_results': results,
                'average_results': avg_results
            }, f, indent=2)

        print("\nAverage Results:")
        print(f"Average Loss: {avg_results['avg_loss']:.4f}")
        print(f"Average Quality Score: {avg_results['avg_quality']:.2f}")
        print("Average Metrics:")
        for metric, score in avg_results['avg_metrics'].items():
            print(f"- {metric}: {score:.2f}")

    except Exception as e:
        print(f"Error in sequence prediction testing: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing Model Generation...")
    test_model_generation()

    print("\nTesting Sequence Prediction...")
    test_sequence_prediction()
