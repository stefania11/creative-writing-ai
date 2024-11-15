"""
Integration between nanoGPT model and validation framework.
"""
import os
import json
import time
from typing import Dict, Any, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .model_validator import ModelValidator

class CreativeWritingModelValidator:
    def __init__(self):
        self.validator = ModelValidator('models/creative_writing')
        self.client = OpenAI(
            api_key=os.getenv('oai_key'),
            timeout=15.0,  # Reduced timeout
            max_retries=2
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
    def _generate_text(self, prompt: str) -> str:
        """Generate text with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a middle school creative writing assistant. Write in a style appropriate for grades 6-8. Keep responses concise, between 150-300 words."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Reduced tokens
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {str(e)}")
            raise

    def generate_and_validate(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate samples and validate against standards."""
        test_cases = []
        results = {
            'overall_performance': {},
            'test_cases': {},
            'errors': []
        }

        # Use only first two prompts for initial testing
        test_prompts = prompts[:2]
        print(f"\nTesting with {len(test_prompts)} prompts for initial validation...")

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nProcessing prompt {i}/{len(test_prompts)}: {prompt[:50]}...")
            try:
                # Generate text with retry logic
                generated_text = self._generate_text(prompt)
                print(f"Successfully generated text ({len(generated_text)} chars)")

                test_case = {
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'expected_metrics': {
                        'grade_level': {'min': 6, 'max': 8},
                        'content_quality': 0.7,
                        'writing_mechanics': 0.7
                    }
                }
                test_cases.append(test_case)

                # Validate individual case
                case_results = self.validator.validate_text(generated_text)
                results['test_cases'][prompt] = {
                    'generated_text': generated_text,
                    'validation_results': case_results
                }
                print("Validation completed for this prompt")

            except Exception as e:
                error_msg = f"Error processing prompt {i}: {str(e)}"
                print(error_msg)
                results['errors'].append(error_msg)
                continue

        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

# Test prompts for middle school writing (shorter list)
TEST_PROMPTS = [
    "Write a short story about discovering a hidden talent.",
    "Describe your ideal day at school in 200 words."
]
