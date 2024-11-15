from setuptools import setup, find_packages

setup(
    name="creative-writing-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "textstat>=0.7.0",
        "nltk>=3.6.0",
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
