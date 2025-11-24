"""
Part 3: Train Counterspeech Generation Model
Trains GPT-2 to generate counterspeech responses to toxic comments
Uses the CONAN (COunter NArratives through Nichesourcing) dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset

# Configuration
SAMPLE_SIZE =20000 # Set to None to use full CONAN dataset, or e.g., 5000 for subset
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 5e-5

class CounterspeechDataset(Dataset):
    """
    Dataset for counterspeech generation
    Format: [TOXIC] <toxic_comment> [RESPONSE] <counterspeech_response>
    """
    
    def __init__(self, toxic_comments, counterspeech_responses, tokenizer, max_length=128):
        self.toxic_comments = toxic_comments
        self.counterspeech = counterspeech_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Special tokens
        self.toxic_token = "[TOXIC]"
        self.response_token = "[RESPONSE]"
        
    def __len__(self):
        return len(self.toxic_comments)
    
    def __getitem__(self, idx):
        toxic = str(self.toxic_comments[idx])
        counter = str(self.counterspeech[idx])
        
        # Create input text: [TOXIC] comment [RESPONSE] counterspeech
        input_text = f"{self.toxic_token} {toxic} {self.response_token} {counter}"
        
        # Tokenize
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        # Mask the toxic comment part - only compute loss on counterspeech generation
        # Find where [RESPONSE] token starts
        response_token_id = self.tokenizer.encode(self.response_token, add_special_tokens=False)[0]
        try:
            response_start = (input_ids == response_token_id).nonzero(as_tuple=True)[0][0].item()
            labels[:response_start+1] = -100  # Don't compute loss on prompt part
        except:
            pass  # If response token not found, use full sequence
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CounterspeechGenerator(nn.Module):
    """
    GPT-2 based counterspeech generation model
    """
    
    def __init__(self, model_name='gpt2-medium'):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.gpt2.config
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate_counterspeech(self, toxic_comment, tokenizer, max_length=100):
        """Generate counterspeech for a toxic comment"""
        prompt = f"[TOXIC] {toxic_comment} [RESPONSE]"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(next(self.parameters()).device)
        
        # Generate
        output = self.gpt2.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract only the counterspeech part
        if "[RESPONSE]" in generated_text:
            counterspeech = generated_text.split("[RESPONSE]")[1].strip()
            # Remove any [TOXIC] tags that might appear
            counterspeech = counterspeech.replace("[TOXIC]", "").strip()
            return counterspeech
        return generated_text


def load_conan_dataset(sample_size=None):
    """
    Load CONAN counterspeech dataset
    Returns hate speech and counterspeech pairs
    """
    print("Loading CONAN dataset from HuggingFace...")
    from datasets import load_dataset
    
    # Load the English version of CONAN
    ds = load_dataset("HiTZ/CONAN-EUS", "en")
    
    print(f"Dataset loaded: {ds}")
    print(f"Available splits: {ds.keys()}")
    
    # CONAN-EUS dataset structure:
    # - 'HS': Hate Speech (the toxic/hate speech comment)
    # - 'CN': Counter Narrative (the counterspeech response)
    # - 'prefix': target group information
    
    # Use train split
    train_data = ds['train']
    
    # Check the first example to understand structure
    if len(train_data) > 0:
        print(f"\nFirst example structure: {train_data[0]}")
        print(f"Available keys: {train_data[0].keys()}")
    
    hate_speech = []
    counterspeech = []
    
    print(f"\nProcessing {len(train_data)} samples from CONAN train split...")
    for example in train_data:
        # Get hate speech and counterspeech using correct field names
        hate = example.get('HS', '')
        counter = example.get('CN', '')
        
        if hate and counter and len(hate.strip()) > 0 and len(counter.strip()) > 0:
            hate_speech.append(hate.strip())
            counterspeech.append(counter.strip())
    
    print(f"✓ Loaded {len(hate_speech)} hate speech-counterspeech pairs from train split")
    
    # Also load validation split if available
    if 'validation' in ds:
        val_data = ds['validation']
        print(f"\nProcessing {len(val_data)} samples from CONAN validation split...")
        
        val_hate = []
        val_counter = []
        
        for example in val_data:
            hate = example.get('HS', '')
            counter = example.get('CN', '')
            
            if hate and counter and len(hate.strip()) > 0 and len(counter.strip()) > 0:
                val_hate.append(hate.strip())
                val_counter.append(counter.strip())
        
        print(f"✓ Loaded {len(val_hate)} validation pairs")
    else:
        # Create our own validation split
        split_idx = int(0.9 * len(hate_speech))
        val_hate = hate_speech[split_idx:]
        val_counter = counterspeech[split_idx:]
        hate_speech = hate_speech[:split_idx]
        counterspeech = counterspeech[:split_idx]
    
    # Sample if requested
    if sample_size is not None and sample_size < len(hate_speech):
        indices = np.random.choice(len(hate_speech), sample_size, replace=False)
        hate_speech = [hate_speech[i] for i in indices]
        counterspeech = [counterspeech[i] for i in indices]
        print(f"\nSampled {sample_size} training pairs")
        
        # Proportionally sample validation
        val_sample_size = int(sample_size * len(val_hate) / (len(hate_speech) + len(val_hate)))
        if val_sample_size > 0 and val_sample_size < len(val_hate):
            val_indices = np.random.choice(len(val_hate), val_sample_size, replace=False)
            val_hate = [val_hate[i] for i in val_indices]
            val_counter = [val_counter[i] for i in val_indices]
            print(f"Sampled {val_sample_size} validation pairs")
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train samples: {len(hate_speech)}")
    print(f"  Validation samples: {len(val_hate)}")
    
    # Print some examples
    print("\n" + "="*70)
    print("Sample CONAN data:")
    print("="*70)
    for i in range(min(3, len(hate_speech))):
        print(f"\nExample {i+1}:")
        print(f"Hate Speech: {hate_speech[i][:150]}...")  # Truncate for display
        print(f"Counterspeech: {counterspeech[i][:150]}...")
    print("="*70 + "\n")
    
    return hate_speech, counterspeech, val_hate, val_counter


def train_counterspeech_model():
    """Main training function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['[TOXIC]', '[RESPONSE]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load CONAN counterspeech dataset
    print("Loading CONAN dataset...")
    train_toxic, train_counter, val_toxic, val_counter = load_conan_dataset(
        sample_size=SAMPLE_SIZE
    )
    
    # Create datasets
    train_dataset = CounterspeechDataset(train_toxic, train_counter, tokenizer, MAX_LENGTH)
    val_dataset = CounterspeechDataset(val_toxic, val_counter, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Initialize model
    print("\nInitializing model...")
    model = CounterspeechGenerator('gpt2-medium')
    
    # Resize embeddings for new special tokens
    model.gpt2.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*70}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_steps
        train_perplexity = np.exp(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_steps
        val_perplexity = np.exp(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        
        print(f"\nTrain Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'tokenizer': tokenizer
            }, 'models/counterspeech_generator_best.pt')
            print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Generate sample counterspeech
        if (epoch + 1) % 2 == 0:
            print("\n" + "="*70)
            print("Sample Counterspeech Generation:")
            print("="*70)
            
            test_toxic_comments = [
                "You're so stupid!",
                "I hate people like you.",
                "Get lost, nobody wants you here."
            ]
            
            for comment in test_toxic_comments:
                response = model.generate_counterspeech(comment, tokenizer, max_length=50)
                print(f"\nToxic: {comment}")
                print(f"Counter: {response}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, 'models/counterspeech_generator_final.pt')
    print("\n✓ Saved final model to models/counterspeech_generator_final.pt")
    
    # Plot training history
    plot_training_history(history)
    
    return model, tokenizer, history


def plot_training_history(history):
    """Plot training curves"""
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Counterspeech Model Training Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity plot
    axes[1].plot(epochs, history['train_perplexity'], label='Train Perplexity', 
                marker='o', linewidth=2)
    axes[1].plot(epochs, history['val_perplexity'], label='Val Perplexity', 
                marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Counterspeech Model Perplexity', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/counterspeech_training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved to outputs/counterspeech_training_history.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("COUNTERSPEECH GENERATION MODEL TRAINING")
    print("="*70)
    
    model, tokenizer, history = train_counterspeech_model()
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)