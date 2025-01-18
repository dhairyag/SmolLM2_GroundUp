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
import torch.nn.functional as F
import time

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
    
    # Keep track of seen examples
    seen_texts = set()
    
    def tokenize_function(examples):
        # Debug print to see what data looks like - reduced frequency
        text_hash = hash(examples["text"][:100])  # Hash first 100 chars
        if text_hash not in seen_texts and len(seen_texts) % 100 == 0:  # Print every 100th new text
            seen_texts.add(text_hash)
            print(f"\nNew unique texts: {len(seen_texts)}")
        else:
            seen_texts.add(text_hash)
            
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        return {
            "input_ids": torch.tensor(outputs["input_ids"]),
            "attention_mask": torch.tensor(outputs["attention_mask"])
        }
    
    # Create training dataloader with buffer_size for proper shuffling
    train_dataset = dataset["train"].shuffle(buffer_size=10000).map(
        tokenize_function, 
        remove_columns=["text"]
    )
    
    print("\nVerifying data streaming...")
    
    # Create the DataLoader
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

def generate_sample_text(model, tokenizer, device, prompt="Once upon a time", max_length=50, temperature=0.7):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        
        for _ in range(max_length):
            outputs, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we predict EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
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
        optimizer = AdamW(model.parameters(), lr=3e-4)
        num_steps = 5000
        checkpoint_interval = 500
        warmup_steps = 100
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Training loop
        model.train()
        step = 0
        train_iterator = iter(train_dataloader)
        last_batch_check = time.time()
        batch_hashes = set()
        
        progress_bar = tqdm(total=num_steps, desc="Training")
        
        print("Starting training...")
        
        while step < num_steps:
            try:
                batch = next(train_iterator)
                
                # Monitor data variety every 5 minutes instead of every minute
                if time.time() - last_batch_check > 300:  # Changed from 60 to 300 seconds
                    batch_hash = hash(str(batch["input_ids"][0][:50].tolist()))
                    batch_hashes.add(batch_hash)
                    print(f"\nUnique batches seen: {len(batch_hashes)}")
                    last_batch_check = time.time()
                    
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
                print("\nRestarted data iterator")
            
            # Learning rate warmup
            if step < warmup_steps:
                lr = 3e-4 * (step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Print shapes for debugging
            if step == 0:
                print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            
            labels = input_ids.clone()
            
            outputs, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if step % 210 == 0:
                print(f"\nStep {step}")
                print(f"Loss: {loss.item():.4f}")
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"Unique batches seen: {len(batch_hashes)}")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Generate sample text every 500 steps with different prompts
            if step % 500 == 0:
                prompts = [
                    "Once upon a time",
                    "The scientific method",
                    "In conclusion,",
                    "The main difference between"
                ]
                for prompt in prompts:
                    sample_text = generate_sample_text(
                        model, 
                        tokenizer, 
                        device, 
                        prompt=prompt,
                        max_length=100,
                        temperature=0.7
                    )
                    print(f"\nPrompt: {prompt}\nGenerated: {sample_text}\n")
            
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