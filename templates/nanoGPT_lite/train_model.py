import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from .transformer_model import CreativeWritingTransformer
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Train the creative writing model and return the checkpoint path"""
    logger.info("Initializing model training...")

    # Model configuration
    config = {
        'vocab_size': 50257,  # GPT-2 default
        'n_embd': 768,
        'n_head': 8,
        'n_layer': 6,
        'dropout': 0.1
    }

    # Initialize model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = CreativeWritingTransformer(**config)
    model = model.to(device)

    # Initialize tokenizer
    tokenizer = model.tokenizer

    # Sample training data (replace with your dataset)
    train_texts = [
        "Write a creative story about a magical library where books come to life at night.",
        "Describe an adventure in a secret garden that only appears during a full moon.",
        "Tell a story about a young inventor who creates a time-traveling bicycle."
    ]

    # Tokenize inputs
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 5

    logger.info("Starting training loop...")
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        _, loss = model(input_ids, input_ids)  # Using input as targets for self-supervised learning

        loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save the model with proper configuration
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/creative_writing_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config,  # Save as model_config instead of config
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer,  # Save tokenizer configuration
        'vocab_size': config['vocab_size'],
        'n_embd': config['n_embd'],
        'n_head': config['n_head'],
        'n_layer': config['n_layer'],
        'dropout': config['dropout']
    }, checkpoint_path)

    logger.info(f"Model saved to {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    train_model()
