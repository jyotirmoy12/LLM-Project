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

SAMPLE_SIZE = 20000 
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 5e-5

class CounterspeechDataset(Dataset):

    def __init__(self, toxic_comments, counterspeech_responses, tokenizer, max_length=128):
        self.toxic_comments = toxic_comments
        self.counterspeech = counterspeech_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.toxic_token = "[TOXIC]"
        self.response_token = "[RESPONSE]"
        
    def __len__(self):
        return len(self.toxic_comments)
    
    def __getitem__(self, idx):
        toxic = str(self.toxic_comments[idx])
        counter = str(self.counterspeech[idx])
        input_text = f"{self.toxic_token} {toxic} {self.response_token} {counter}"
        
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        labels = input_ids.clone()
        response_token_id = self.tokenizer.encode(self.response_token, add_special_tokens=False)[0]
        response_start = (input_ids == response_token_id).nonzero(as_tuple=True)[0][0].item()
        labels[:response_start+1] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CounterspeechGenerator(nn.Module):

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
        prompt = f"[TOXIC] {toxic_comment} [RESPONSE]"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(next(self.parameters()).device)
        
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
        
        if "[RESPONSE]" in generated_text:
            counterspeech = generated_text.split("[RESPONSE]")[1].strip()
            counterspeech = counterspeech.replace("[TOXIC]", "").strip()
            return counterspeech
        return generated_text


def load_conan_dataset(sample_size=None):
    ds = load_dataset("HiTZ/CONAN-EUS", "en")
    train_data = ds['train']
    
    hate_speech = []
    counterspeech = []

    for example in train_data:
        hate = example.get('HS', '')
        counter = example.get('CN', '')
        
        if hate and counter and len(hate.strip()) > 0 and len(counter.strip()) > 0:
            hate_speech.append(hate.strip())
            counterspeech.append(counter.strip())

    val_hate = []
    val_counter = []
    
    if 'validation' in ds:
        val_data = ds['validation']

        for example in val_data:
            hate = example.get('HS', '')
            counter = example.get('CN', '')
            
            if hate and counter and len(hate.strip()) > 0 and len(counter.strip()) > 0:
                val_hate.append(hate.strip())
                val_counter.append(counter.strip())

    if sample_size is not None and sample_size < len(hate_speech):
        original_train_len = len(hate_speech)
        indices = np.random.choice(original_train_len, sample_size, replace=False)
        hate_speech = [hate_speech[i] for i in indices]
        counterspeech = [counterspeech[i] for i in indices]
        
        val_sample_size = int(sample_size * len(val_hate) / (original_train_len + len(val_hate)))
        if val_sample_size > 0 and val_sample_size < len(val_hate):
            val_indices = np.random.choice(len(val_hate), val_sample_size, replace=False)
            val_hate = [val_hate[i] for i in val_indices]
            val_counter = [val_counter[i] for i in val_indices]
    
    return hate_speech, counterspeech, val_hate, val_counter


def train_counterspeech_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'additional_special_tokens': ['[TOXIC]', '[RESPONSE]']}
    tokenizer.add_special_tokens(special_tokens)
    
    train_toxic, train_counter, val_toxic, val_counter = load_conan_dataset(sample_size=SAMPLE_SIZE)
    
    train_dataset = CounterspeechDataset(train_toxic, train_counter, tokenizer, MAX_LENGTH)
    val_dataset = CounterspeechDataset(val_toxic, val_counter, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CounterspeechGenerator('gpt2-medium')
    model.gpt2.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
        
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_steps
        val_perplexity = np.exp(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        
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
        
        if (epoch + 1) % 2 == 0:
            test_toxic_comments = [
                "You're so stupid!",
                "I hate people like you.",
                "Get lost, nobody wants you here."
            ]
            
            for comment in test_toxic_comments:
                response = model.generate_counterspeech(comment, tokenizer, max_length=50)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, 'models/counterspeech_generator_final.pt')
    
    plot_training_history(history)
    
    return model, tokenizer, history


def plot_training_history(history):
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Counterspeech Model Training Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_perplexity'], label='Train Perplexity', marker='o', linewidth=2)
    axes[1].plot(epochs, history['val_perplexity'], label='Val Perplexity', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Counterspeech Model Perplexity', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/counterspeech_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    model, tokenizer, history = train_counterspeech_model()