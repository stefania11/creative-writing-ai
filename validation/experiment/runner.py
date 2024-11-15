"""
Validation experiment runner for creative writing model.
"""
import os
import json
from typing import Dict, Any
from ..validation_pipeline import ValidationPipeline

class ValidationExperiment:
    def __init__(self, config_path: str):
        self.validator = ValidationPipeline()
        self.config = self._load_config(config_path)
        self.results_dir = os.path.join('results', 'validation')
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        if not os.path.exists(config_path):
            return self._create_default_config(config_path)
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_default_config(self, config_path: str) -> Dict[str, Any]:
        """Create default configuration for middle school standards."""
        config = {
            'grade_level': {'min': 6, 'max': 8},
            'evaluation_criteria': {
                'content_quality': 0.7,
                'writing_mechanics': 0.7,
                'creativity': 0.6
            },
            'test_prompts': [
                "Write a story about discovering an unexpected talent.",
                "Describe your perfect day at school."
            ]
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config

    def run_experiment(self) -> Dict[str, Any]:
        """Run validation experiment."""
        results = {}
        for prompt in self.config['test_prompts']:
            print(f"\nValidating prompt: {prompt}")
            sample_text = "This is a placeholder for generated text."
            validation_results = self.validator.validate_text(sample_text)
            results[prompt] = {
                'prompt': prompt,
                'validation_results': validation_results,
                'meets_standards': self._check_standards(validation_results)
            }
        return results
