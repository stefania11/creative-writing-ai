"""
Comprehensive model performance verification for middle school creative writing.
"""
import os
import json
import openai
from typing import Dict, Any
from validation.validation_pipeline import ValidationPipeline
from validation.experiment_runner import ValidationExperiment

class ModelPerformanceVerifier:
    def __init__(self):
        self.validator = ValidationPipeline()
        self.experiment = ValidationExperiment('templates/nanoGPT_lite/ideas/validation_experiment.json')
        openai.api_key = os.getenv('oai_key')

    def generate_test_samples(self, prompts: list) -> Dict[str, str]:
        """Generate test samples using OpenAI API."""
        samples = {}
        for prompt in prompts:
            print(f"Generating sample for prompt: {prompt}")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a middle school creative writing assistant. Write in a style appropriate for grades 6-8."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            samples[prompt] = response.choices[0].message.content
        return samples

    def verify_performance(self, output_dir: str = 'results') -> Dict[str, Any]:
        """Run comprehensive performance verification."""
        os.makedirs(output_dir, exist_ok=True)

        test_prompts = [
            "Write a story about discovering an unexpected talent.",
            "Describe your perfect day at school.",
            "Write about a time when you had to solve a difficult problem.",
            "Create a story about making a new friend in an unusual way."
        ]

        print("Starting performance verification...")
        print("Generating test samples...")
        samples = self.generate_test_samples(test_prompts)

        results = {}
        for prompt, text in samples.items():
            print(f"\nValidating response for prompt: {prompt}")
            validation_results = self.validator.validate_text(text)

            report = {
                'prompt': prompt,
                'generated_text': text,
                'validation_results': validation_results,
                'meets_standards': self._check_standards(validation_results)
            }
            results[prompt] = report

        output_path = os.path.join(output_dir, 'performance_verification.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)
        return results

    def _check_standards(self, results: Dict[str, Any]) -> bool:
        """Check if results meet middle school standards."""
        grade_level = results['grade_level_assessment']['readability']
        grade_appropriate = 6 <= grade_level <= 8

        content_scores = results['content_analysis']
        content_quality = all(score >= 0.7 for score in content_scores.values())

        writing_scores = results['writing_evaluation']
        writing_quality = all(
            score >= 0.7 for scores in writing_scores.values()
            for score in scores.values()
        )

        return grade_appropriate and content_quality and writing_quality

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of verification results."""
        print("\n=== Performance Verification Summary ===")
        total_samples = len(results)
        passing_samples = sum(1 for result in results.values() if result['meets_standards'])

        print(f"\nTotal Samples: {total_samples}")
        print(f"Passing Samples: {passing_samples}")
        print(f"Success Rate: {(passing_samples/total_samples)*100:.1f}%")

        print("\nDetailed Results:")
        for prompt, result in results.items():
            print(f"\nPrompt: {prompt[:50]}...")
            print(f"Meets Standards: {result['meets_standards']}")
            print("Key Metrics:")
            for category, scores in result['validation_results'].items():
                if isinstance(scores, dict):
                    avg_score = sum(scores.values()) / len(scores)
                    print(f"- {category}: {avg_score:.2f}")

if __name__ == "__main__":
    verifier = ModelPerformanceVerifier()
    verifier.verify_performance()
