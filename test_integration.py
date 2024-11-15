import torch
import torch.nn.functional as F
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
    print("\nStarting sequence prediction testing...")

    # Load test data
    try:
        with open('data/samples/test_data.json', 'r') as f:
            test_data = json.load(f)
            test_samples = test_data['samples']
            print(f"Loaded {len(test_samples)} test samples")
    except FileNotFoundError:
        print("Test data not found. Please ensure test_data.json exists in data/samples/")
        return
    except KeyError:
        print("Invalid test data format. Expected 'samples' key in test_data.json")
        return
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return

    # Initialize model and validator
    try:
        model_config = {
            'vocab_size': 50257,
            'n_embd': 256,
            'n_head': 8,
            'n_layer': 4,
            'dropout': 0.1
        }
        print("Initializing model with config:", model_config)

        model = CreativeWritingTransformer(**model_config)
        validator = SequenceValidator()
        print("Model and validator initialized successfully")
    except Exception as e:
        print(f"Error initializing model or validator: {str(e)}")
        return

    # Test sequence prediction
    try:
        results = []
        for i, sample in enumerate(test_samples, 1):
            print(f"\nProcessing sample {i}/{len(test_samples)}")

            try:
                # Convert sample data to tensors
                input_tensor = torch.tensor(sample['input_ids']).unsqueeze(0)
                target_tensor = torch.tensor(sample['target_ids']).unsqueeze(0)

                # Get model predictions
                with torch.no_grad():
                    output = model(input_tensor, target_tensor)
                    if isinstance(output, tuple):
                        logits, loss = output
                    else:
                        logits = output
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            target_tensor.view(-1)
                        )

                    # Calculate prediction accuracy
                    pred_tokens = torch.argmax(logits, dim=-1)
                    accuracy = (pred_tokens == target_tensor).float().mean().item()

                # Decode predictions for comparison
                tokenizer = model.get_tokenizer()
                pred_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)

                # Evaluate quality
                quality_score = validator.evaluate_completion_quality(sample['target'])
                pred_quality = validator.evaluate_completion_quality(pred_text)

                result = {
                    'prompt': sample['prompt'],
                    'target': sample['target'],
                    'prediction': pred_text,
                    'loss': float(loss.item()),
                    'accuracy': accuracy,
                    'quality_scores': {
                        'target': quality_score,
                        'prediction': pred_quality
                    },
                    'metrics': {
                        'target': {
                            'sentence_structure': validator._check_sentence_structure(sample['target']),
                            'vocabulary_level': validator._check_vocabulary_level(sample['target']),
                            'coherence': validator._check_coherence(sample['target'])
                        },
                        'prediction': {
                            'sentence_structure': validator._check_sentence_structure(pred_text),
                            'vocabulary_level': validator._check_vocabulary_level(pred_text),
                            'coherence': validator._check_coherence(pred_text)
                        }
                    }
                }
                results.append(result)

                # Print sample results
                print(f"Sample {i} Results:")
                print(f"Prompt: {sample['prompt']}")
                print(f"Target: {sample['target']}")
                print(f"Prediction: {pred_text}")
                print(f"Loss: {result['loss']:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Quality Scores - Target: {quality_score:.2f}, Prediction: {pred_quality:.2f}")

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        if not results:
            print("No successful predictions to analyze")
            return

        # Calculate averages
        avg_results = {
            'avg_loss': float(np.mean([r['loss'] for r in results])),
            'avg_accuracy': float(np.mean([r['accuracy'] for r in results])),
            'avg_quality': {
                'target': float(np.mean([r['quality_scores']['target'] for r in results])),
                'prediction': float(np.mean([r['quality_scores']['prediction'] for r in results]))
            },
            'avg_metrics': {
                'target': {
                    metric: float(np.mean([r['metrics']['target'][metric] for r in results]))
                    for metric in results[0]['metrics']['target']
                },
                'prediction': {
                    metric: float(np.mean([r['metrics']['prediction'][metric] for r in results]))
                    for metric in results[0]['metrics']['prediction']
                }
            }
        }

        # Print summary
        print("\nTest Summary:")
        print(f"Processed {len(results)}/{len(test_samples)} samples successfully")
        print(f"Average Loss: {avg_results['avg_loss']:.4f}")
        print(f"Average Accuracy: {avg_results['avg_accuracy']:.4f}")
        print("Average Quality Scores:")
        print(f"- Target: {avg_results['avg_quality']['target']:.2f}")
        print(f"- Prediction: {avg_results['avg_quality']['prediction']:.2f}")
        print("Average Metrics:")
        for category in ['target', 'prediction']:
            print(f"\n{category.title()} Metrics:")
            for metric, score in avg_results['avg_metrics'][category].items():
                print(f"- {metric}: {score:.2f}")

        # Save results
        with open('sequence_prediction_results.json', 'w') as f:
            json.dump({
                'individual_results': results,
                'average_results': avg_results
            }, f, indent=2)
        print("\nResults saved to sequence_prediction_results.json")

    except Exception as e:
        print(f"Error in sequence prediction testing: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing Model Generation...")
    test_model_generation()

    print("\nTesting Sequence Prediction...")
    test_sequence_prediction()
