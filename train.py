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
import json

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

def load_and_prepare_data(tokenizer, batch_size=32):
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        streaming=True
    )
    
    # Keep track of seen examples with more detail
    seen_texts = {}  # hash -> first few words
    
    def tokenize_function(examples):
        # Handle batched inputs
        texts = examples["text"]
        
        # Process each text in the batch
        for text in texts:
            if isinstance(text, str):  # Make sure it's a string
                text_hash = hash(text[:100])
                if text_hash not in seen_texts:
                    seen_texts[text_hash] = text[:50]
                    #if len(seen_texts) % 100 == 0:
                        #print(f"\nNew unique texts: {len(seen_texts)}")
                        # Print the most recent text
                        #print(f"Recent sample: {text[:100]}...")
        
        # Tokenize all texts in batch
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(outputs["input_ids"]),
            "attention_mask": torch.tensor(outputs["attention_mask"])
        }
    
    # Create training dataloader with proper streaming setup
    train_dataset = (
        dataset["train"]
        .shuffle(
            buffer_size=10000,  # Buffer for shuffling
            seed=42  # For reproducibility
        )
        .map(
            tokenize_function,
            remove_columns=["text"],
            batched=True,
            batch_size=100  # Internal batch size for mapping
        )
    )
    
    print("\nVerifying data streaming...")
    
    # Create the DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle again as we're already shuffling in the pipeline
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
    
    # Add debug prints
    print(f"\nGenerating with temperature: {temperature}")
    
    with torch.no_grad():
        input_ids = inputs["input_ids"]  # Shape: [1, seq_len]
        attention_mask = torch.ones_like(input_ids)  # Shape: [1, seq_len]
        
        generated_tokens = []
        for _ in range(max_length):
            outputs, _ = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Filter out special tokens
            for special_token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
                if special_token_id is not None:
                    next_token_logits[:, special_token_id] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from top-k
            top_k = 40
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            next_token = top_k_indices[0][torch.multinomial(top_k_probs[0], num_samples=1)]
            
            # Reshape next_token to match input_ids dimensions [1, 1]
            next_token = next_token.view(1, 1)
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)  # Now both tensors are [1, seq_len]
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    model.train()
    return f"{prompt} {generated_text}"

def save_metrics(metrics_dict, filename="training_metrics.pt"):
    """Save training metrics to a file"""
    torch.save(metrics_dict, filename)
    print(f"Metrics saved to {filename}")

def save_model_for_hf(model, tokenizer, output_dir):
    """Save model and tokenizer in a HuggingFace compatible format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model.config.__dict__, f)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Model and tokenizer saved to {output_dir}")

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
        unique_content = set()  # Track unique content
        
        progress_bar = tqdm(total=num_steps, desc="Training")
        
        print("Starting training...")
        
        # Initialize metrics dictionary
        metrics = {
            'steps': [],
            'loss': [],
            'learning_rate': [],
            'unique_texts': [],
            'timestamp': [],
            'unique_content_count': []
        }
        
        while step < num_steps:
            try:
                batch = next(train_iterator)
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                
                # Calculate loss first
                outputs, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                
                # Record metrics
                metrics['steps'].append(step)
                metrics['loss'].append(loss.item())
                metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
                metrics['timestamp'].append(time.time())
                metrics['unique_content_count'].append(len(unique_content))
                
                # Now we can use loss in our prints and checks
                if step % 100 == 0:
                    print(f"\nStep {step}")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                    print(f"Unique content: {len(unique_content)}")
                    
                    # Save metrics periodically
                    save_metrics(metrics, f"checkpoints/metrics_{step}.pt")
                    
                
                # Track unique content
                for seq in batch["input_ids"]:
                    content = tokenizer.decode(seq[:50])
                    unique_content.add(content)
                
                # if time.time() - last_batch_check > 300:
                #     print(f"\nUnique content pieces seen: {len(unique_content)}")
                #     sample = list(unique_content)[-1]
                #     #print(f"Latest unique content: {sample[:100]}...")
                #     last_batch_check = time.time()
                
                # Gradient steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Learning rate warmup
                if step < warmup_steps:
                    lr = 3e-4 * (step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                # Generate samples and save checkpoints
                if step % 500 == 0 and step > 0:
                    for prompt in ["Once upon a time", "The scientific method"]:
                        sample_text = generate_sample_text(
                            model, tokenizer, device, 
                            prompt=prompt, 
                            max_length=50,
                            temperature=0.8
                        )
                        print(f"\nGenerated ({step} steps):\n<< {sample_text} >>")
                    
                    save_checkpoint(model, optimizer, step, f"checkpoints/checkpoint_{step}.pt")
                    # Save model in HuggingFace format
                    save_model_for_hf(model, tokenizer, f"smollm2_model_step_{step}")
                
                step += 1
                progress_bar.update(1)
                
            except StopIteration:
                print("\nRestarting data iterator...")
                train_iterator = iter(train_dataloader)
                continue
            
            except Exception as e:
                # Save metrics even if training fails
                save_metrics(metrics, "checkpoints/metrics_interrupted.pt")
                print(f"Error in training loop: {str(e)}")
                raise e
        
        # Save final metrics
        save_metrics(metrics, "checkpoints/metrics_final.pt")
        
        # Save final checkpoint
        save_checkpoint(model, optimizer, step, "checkpoints/checkpoint_final.pt")
        save_model_for_hf(model, tokenizer, "smollm2_model_final")
        
        print("\nStarting additional training...")
        # Additional 50 steps after loading checkpoint
        print("\nLoading final checkpoint and training for 50 more steps...")
        load_checkpoint(model, optimizer, "checkpoints/checkpoint_final.pt")

        # Additional training metrics
        additional_metrics = {
            'steps': [],
            'loss': [],
            'timestamp': []
        }
        
        # Additional 50 steps after loading checkpoint
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
            
            # Record additional metrics
            additional_metrics['steps'].append(num_steps + extra_step)
            additional_metrics['loss'].append(loss.item())
            additional_metrics['timestamp'].append(time.time())
        
        # Save additional metrics
        save_metrics(additional_metrics, "checkpoints/metrics_additional.pt")
        
        # Save final model after additional training
        save_checkpoint(model, optimizer, num_steps + 50, "checkpoints/checkpoint_final_extended.pt")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 