"""
Integration between nanoGPT model and validation framework.
"""
import os
import json
import torch
from typing import Dict, Any, List
import openai
from .model_validator import ModelValidator

class CreativeWritingModelValidator:
    def __init__(self):
        self.validator = ModelValidator('models/creative_writing')
        self.api_key = os.getenv('oai_key')
        openai.api_key = self.api_key

    def generate_and_validate(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate samples and validate against standards."""
        test_cases = []
        for prompt in prompts:
            # Generate text using OpenAI for now (will replace with local model)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a middle school creative writing assistant. Write in a style appropriate for grades 6-8."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            generated_text = response.choices[0].message.content

            test_cases.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'expected_metrics': {
                    'grade_level': {'min': 6, 'max': 8},
                    'content_quality': 0.7,
                    'writing_mechanics': 0.7
                }
            })

        # Run validation
        results = self.validator.validate_model(test_cases)

        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

# Test prompts for middle school writing
TEST_PROMPTS = [
    "Write a story about discovering an unexpected talent.",
    "Describe your perfect day at school.",
    "Write about a time when you had to solve a difficult problem.",
    "Create a story about making a new friend in an unusual way."
]
