import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from model import SmolLM2Config, SmolLM2ForCausalLM
import os
from tqdm import tqdm
import random
from huggingface_hub import login

def setup_training():
    # Login to Hugging Face using environment variable
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
    else:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        
    # Initialize wandb
    #wandb.init(project="smollm2-training")
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize model and config
    config = SmolLM2Config()
    model = SmolLM2ForCausalLM(config)
    # Print model info and parameter count
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def load_and_prepare_data(tokenizer, batch_size=4):
    # Use a public dataset without authentication, specifying the config
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True)
    
    def tokenize_function(examples):
        # Remove return_tensors="pt" as DataLoader will handle tensor conversion
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # Convert to tensors manually to ensure proper shape
        return {
            "input_ids": torch.tensor(outputs["input_ids"]),
            "attention_mask": torch.tensor(outputs["attention_mask"])
        }
    
    # Create training dataloader
    train_dataset = dataset["train"].map(tokenize_function, remove_columns=["text"])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return train_dataloader

def save_checkpoint(model, optimizer, step, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def generate_sample_text(model, tokenizer, device, prompt="Once upon a time", max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_sequences = []
        input_ids = inputs["input_ids"]
        
        for _ in range(max_length):
            outputs, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text

def main():
    try:
        # Setup
        model, tokenizer, device = setup_training()
        print(f"Using device: {device}")
        
        train_dataloader = load_and_prepare_data(tokenizer)
        print("Data loading successful")
        
        # Training parameters
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_steps = 5000
        checkpoint_interval = 500
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Training loop
        model.train()
        step = 0
        train_iterator = iter(train_dataloader)
        
        progress_bar = tqdm(total=num_steps, desc="Training")
        
        print("Starting training...")
        
        while step < num_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            # Move batch to device and ensure proper shapes
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Print shapes for debugging
            if step == 0:
                print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            
            labels = input_ids.clone()
            
            outputs, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            if step % 10 == 0:  # Print loss more frequently
                print(f"\nStep {step}, Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Generate sample text every 500 steps
            if step % 500 == 0:
                sample_text = generate_sample_text(model, tokenizer, device)
                print(f"\nStep {step} - Generated text: {sample_text}")
            
            # Save checkpoint
            if step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step, f"checkpoints/checkpoint_{step}.pt")
            
            step += 1
            progress_bar.update(1)
        
        # Save final checkpoint
        save_checkpoint(model, optimizer, step, "checkpoints/checkpoint_final.pt")
        
        print("\nStarting additional training...")
        # Additional 50 steps after loading checkpoint
        print("\nLoading final checkpoint and training for 50 more steps...")
        load_checkpoint(model, optimizer, "checkpoints/checkpoint_final.pt")
        
        for extra_step in range(50):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            
            outputs, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            print(f"Additional step {extra_step}, Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save final model after additional training
        save_checkpoint(model, optimizer, num_steps + 50, "checkpoints/checkpoint_final_extended.pt")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 