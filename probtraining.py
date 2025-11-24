import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration
SAMPLE_SIZE = 10000  # Set to None to use full dataset

# Load preprocessed CSVs
tr_fp = 'Dataset/train_preprocessed.csv'
val_fp = 'Dataset/val_preprocessed.csv'
te_fp = 'Dataset/test_preprocessed.csv'

tr_df = pd.read_csv(tr_fp)
val_df = pd.read_csv(val_fp)
te_df = pd.read_csv(te_fp)

# Apply sampling if specified
if SAMPLE_SIZE is not None:
    tr_df = tr_df.sample(n=min(SAMPLE_SIZE, len(tr_df)), random_state=42)
    val_df = val_df.sample(n=min(int(SAMPLE_SIZE*0.1), len(val_df)), random_state=42)
    te_df = te_df.sample(n=min(int(SAMPLE_SIZE*0.1), len(te_df)), random_state=42)
    print(f"Using sample: Train={len(tr_df)}, Val={len(val_df)}, Test={len(te_df)}")

# Dataset class
class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        y = self.labels[idx]
        
        enc = self.tok(
            txt,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(y, dtype=torch.long)
        }

# Initialize tokenizer
tok = GPT2Tokenizer.from_pretrained('gpt2-medium')
tok.pad_token = tok.eos_token

max_len = 128
bs = 16

# Create datasets with sampled data
train_ds = ToxicCommentDataset(tr_df['comment_text'].values,
                               tr_df['is_toxic'].values, tok, max_len)
val_ds = ToxicCommentDataset(val_df['comment_text'].values,
                             val_df['is_toxic'].values, tok, max_len)
test_ds = ToxicCommentDataset(te_df['comment_text'].values,
                              te_df['is_toxic'].values, tok, max_len)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

# Model: linear toxic probe
class ToxicProbeModel(nn.Module):

    def __init__(self, model_name='gpt2-medium', freeze_gpt2=True):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hsz = self.gpt2.config.hidden_size
        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False
        self.probe = nn.Linear(self.hsz, 2)  # binary

    def forward(self, input_ids, attention_mask):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B, T, H)
        mask_exp = attention_mask.unsqueeze(-1).expand(hs.size()).float()
        sum_h = torch.sum(hs * mask_exp, dim=1)
        sum_m = torch.clamp(mask_exp.sum(1), min=1e-9)
        avg_h = sum_h / sum_m  # (B, H)   --> xÌ„^(L-1)
        logits = self.probe(avg_h)
        return logits, avg_h

# Eval helper

def evaluate(mdl, dl, crit, dev):
    mdl.eval()
    tot_loss = 0.0
    all_p = []
    all_pred = []
    all_lbl = []
    with torch.no_grad():
        for b in tqdm(dl, desc="Eval", leave=False):
            ids = b['input_ids'].to(dev)
            am = b['attention_mask'].to(dev)
            y = b['labels'].long().to(dev)
            logits, _ = mdl(ids, am)
            loss = crit(logits, y)
            tot_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_p.extend(probs.tolist())
            all_pred.extend(preds.tolist())
            all_lbl.extend(y.cpu().numpy().tolist())
    avg_loss = tot_loss / len(dl)
    acc = accuracy_score(all_lbl, all_pred)
    auc = roc_auc_score(all_lbl, all_p)
    return avg_loss, acc, auc, all_pred, all_lbl

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 20
lr = 1e-4

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
# Init model, loss, opt
mdl = ToxicProbeModel('gpt2-medium', freeze_gpt2=True)
mdl = mdl.to(dev)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(mdl.probe.parameters(), lr=lr)

# Training loop
best_val_auc = 0.0
history = {'tr_loss':[], 'tr_acc':[], 'tr_auc':[], 'val_loss':[], 'val_acc':[], 'val_auc':[]}

for ep in range(epochs):
    print(f"Epoch {ep+1}/{epochs}")
    mdl.train()
    running_loss = 0.0
    tr_preds = []
    tr_lbls = []
    tr_probs = []
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for b in pbar:
        ids = b['input_ids'].to(dev)
        am = b['attention_mask'].to(dev)
        y = b['labels'].long().to(dev)

        opt.zero_grad()
        logits, _ = mdl(ids, am)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        tr_probs.extend(probs.tolist())
        tr_preds.extend(preds.tolist())
        tr_lbls.extend(y.cpu().numpy().tolist())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    tr_loss = running_loss / len(train_loader)
    tr_acc = accuracy_score(tr_lbls, tr_preds)
    tr_auc = roc_auc_score(tr_lbls, tr_probs)

    # validate
    val_loss, val_acc, val_auc, val_preds, val_lbls = evaluate(mdl, val_loader, crit, dev)

    history['tr_loss'].append(tr_loss); history['tr_acc'].append(tr_acc); history['tr_auc'].append(tr_auc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_auc'].append(val_auc)

    print(f"Train - loss:{tr_loss:.4f} acc:{tr_acc:.4f} auc:{tr_auc:.4f}")
    print(f"Val   - loss:{val_loss:.4f} acc:{val_acc:.4f} auc:{val_auc:.4f}")

    # save best
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': mdl.state_dict(),
            'probe_weights': mdl.probe.weight.data.cpu().numpy(),
            'probe_bias': mdl.probe.bias.data.cpu().numpy()
        }, 'models/toxic_probe_best.pt')

    # classification report on last epoch
    if ep == epochs - 1:
        print(classification_report(val_lbls, val_preds, target_names=['Non-Toxic','Toxic']))

test_loss, test_acc, test_auc, test_preds, test_lbls = evaluate(mdl, test_loader, crit, dev)
print(f"Test - loss:{test_loss:.4f} acc:{test_acc:.4f} auc:{test_auc:.4f}")
print(classification_report(test_lbls, test_preds, target_names=['Non-Toxic','Toxic']))

# Save final model & plot history
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': mdl.state_dict(),
    'probe_weights': mdl.probe.weight.data.cpu().numpy(),
    'probe_bias': mdl.probe.bias.data.cpu().numpy()
}, 'models/toxic_probe_final.pt')
print("Saved final model: models/toxic_probe_final.pt")

# plotting 
os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss plot
axes[0].plot(history['tr_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history['tr_acc'], label='Train Acc', marker='o')
axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUC plot
axes[2].plot(history['tr_auc'], label='Train AUC', marker='o')
axes[2].plot(history['val_auc'], label='Val AUC', marker='s')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].set_title('Training and Validation AUC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
plt.close()