
import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ToxicVectorComparator:
 
    def __init__(self, 
                 base_model_name='gpt2-medium',
                 probe_path='models/toxic_probe_final.pt',
                 counterspeech_path='models/counterspeech_generator_final.pt'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        probe_checkpoint = torch.load(probe_path, map_location=self.device)
        self.w_toxic = probe_checkpoint['probe_weights'][1]  # Shape: (hidden_size,)
        print(f"W_toxic shape: {self.w_toxic.shape}")
        
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name).to(self.device)
        self.base_model.eval()
        
        cs_checkpoint = torch.load(counterspeech_path, map_location=self.device)

        # Initialize tokenizer with special tokens (same as training)
        from transformers import GPT2Tokenizer
        temp_tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        temp_tokenizer.pad_token = temp_tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['[TOXIC]', '[RESPONSE]']}
        temp_tokenizer.add_special_tokens(special_tokens)

        # Load model and resize embeddings to match training
        self.cs_model = GPT2LMHeadModel.from_pretrained(base_model_name).to(self.device)
        self.cs_model.resize_token_embeddings(len(temp_tokenizer))  # THIS IS THE KEY LINE

        # Now load the state dict
        cs_state_dict = cs_checkpoint['model_state_dict']
        
        # Load the fine-tuned weights
        cs_state_dict = cs_checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in cs_state_dict.items():
            if key.startswith('gpt2.'):
                new_key = key[5:] 
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.cs_model.load_state_dict(new_state_dict, strict=False)
        self.cs_model.eval()
        
        self.config = self.base_model.config
        self.n_layers = self.config.n_layer
        self.hidden_size = self.config.n_embd
        
        print(f"Model config: {self.n_layers} layers, hidden_size={self.hidden_size}")
    
    def extract_mlp_value_vectors(self, model, model_name="model"):

        value_vectors = {}
        
        for layer_idx in tqdm(range(self.n_layers), desc=f"Extracting {model_name}"):
            # Access MLP output projection weights
            mlp = model.transformer.h[layer_idx].mlp
            w_proj = mlp.c_proj.weight.data.cpu().numpy()  # Shape: (hidden_size, mlp_size)
            
            # Each row is a value vector
            for neuron_idx in range(w_proj.shape[1]):
                value_vector = w_proj[:, neuron_idx]
                value_vectors[(layer_idx, neuron_idx)] = value_vector
        
        print(f"Extracted {len(value_vectors)} value vectors from {model_name}")
        return value_vectors
    
    def compute_toxic_similarities(self, value_vectors, model_name="model"):
        similarities = {}
        
        for (layer, idx), vector in tqdm(value_vectors.items(), desc="Computing similarities"):
            sim = cosine_similarity(
                self.w_toxic.reshape(1, -1),
                vector.reshape(1, -1)
            )[0, 0]
            similarities[(layer, idx)] = sim
        
        return similarities
    
    def compute_overlap_metrics(self, base_sims, cs_sims, top_k=128):

        # Get top-k vectors from each model
        base_sorted = sorted(base_sims.items(), key=lambda x: x[1], reverse=True)
        cs_sorted = sorted(cs_sims.items(), key=lambda x: x[1], reverse=True)
        
        base_top_k = set([key for key, _ in base_sorted[:top_k]])
        cs_top_k = set([key for key, _ in cs_sorted[:top_k]])
        
        # Compute overlap
        overlap = base_top_k & cs_top_k
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / top_k) * 100
        
        # Jaccard similarity
        union = base_top_k | cs_top_k
        jaccard = overlap_count / len(union)
        
        print(f"Top-{top_k} overlap: {overlap_count}/{top_k} ({overlap_percentage:.2f}%)")
        print(f"Jaccard similarity: {jaccard:.4f}")
        
        return {
            'top_k': top_k,
            'overlap_count': overlap_count,
            'overlap_percentage': overlap_percentage,
            'jaccard': jaccard,
            'base_top_k': base_top_k,
            'cs_top_k': cs_top_k,
            'overlap_set': overlap
        }
    
    def compute_correlation_metrics(self, base_sims, cs_sims):
        
        # Align the dictionaries (same keys)
        common_keys = set(base_sims.keys()) & set(cs_sims.keys())
        
        base_vals = [base_sims[k] for k in common_keys]
        cs_vals = [cs_sims[k] for k in common_keys]
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(base_vals, cs_vals)
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = spearmanr(base_vals, cs_vals)
        
        print(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_vectors': len(common_keys)
        }
    
    
    
    def run(self):
        
        # Extract vectors from both models
        base_vectors = self.extract_mlp_value_vectors(self.base_model, "base model")
        cs_vectors = self.extract_mlp_value_vectors(self.cs_model, "counterspeech model")
        
        # Compute similarities
        base_sims = self.compute_toxic_similarities(base_vectors, "base model")
        cs_sims = self.compute_toxic_similarities(cs_vectors, "counterspeech model")
        
        # Overlap analysis
        overlap_results = {}
        for k in [32, 64, 128, 256]:
            overlap_results[k] = self.compute_overlap_metrics(base_sims, cs_sims, top_k=k)
        
        # Correlation analysis
        correlation_results = self.compute_correlation_metrics(base_sims, cs_sims)
    
        
        # Store all results
        results = {
            'base_similarities': base_sims,
            'cs_similarities': cs_sims,
            'overlap': overlap_results,
            'correlation': correlation_results,
        }
        
        return results
    
    

    def plot_correlation_scatter(self, base_sims, cs_sims, corr_results, save_path):
    
        # Get aligned values
        common_keys = set(base_sims.keys()) & set(cs_sims.keys())
        base_vals = [base_sims[k] for k in common_keys]
        cs_vals = [cs_sims[k] for k in common_keys]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Hexbin plot for density
        hexbin = ax.hexbin(base_vals, cs_vals, gridsize=50, cmap='YlOrRd', mincnt=1)
        
        # Add diagonal line (perfect correlation)
        min_val = min(min(base_vals), min(cs_vals))
        max_val = max(max(base_vals), max(cs_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect Correlation')
        
        ax.set_xlabel('Base Model Similarity', fontsize=12)
        ax.set_ylabel('Counterspeech Model Similarity', fontsize=12)
        ax.set_title('Toxic Vector Similarity Correlation\nBase vs Counterspeech Model', 
                    fontsize=14, fontweight='bold')
        
        # Add correlation statistics
        textstr = f"Pearson r: {corr_results['pearson_r']:.4f}\n"
        textstr += f"Spearman ρ: {corr_results['spearman_r']:.4f}\n"
        textstr += f"n = {corr_results['n_vectors']:,} vectors"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Count', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_overlap_analysis(self, overlap_results, save_path):
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        top_ks = sorted(overlap_results.keys())
        overlaps = [overlap_results[k]['overlap_percentage'] for k in top_ks]
        jaccards = [overlap_results[k]['jaccard'] * 100 for k in top_ks]
        
        # Plot 1: Overlap percentage
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(top_ks)), overlaps, color='#4472C4', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(top_ks)))
        ax1.set_xticklabels([f'Top-{k}' for k in top_ks])
        ax1.set_ylabel('Overlap Percentage (%)', fontsize=11)
        ax1.set_title('Toxic Vector Overlap Between Models', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars1, overlaps):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Jaccard similarity
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(top_ks)), jaccards, color='#ED7D31', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(top_ks)))
        ax2.set_xticklabels([f'Top-{k}' for k in top_ks])
        ax2.set_ylabel('Jaccard Similarity (%)', fontsize=11)
        ax2.set_title('Jaccard Similarity Between Top Toxic Vectors', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars2, jaccards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
def main():
    
    # Initialize comparator
    comparator = ToxicVectorComparator(
        base_model_name='gpt2-medium',
        probe_path='models/toxic_probe_final.pt',
        counterspeech_path='models/counterspeech_generator_final.pt'
    )

    
    
    # Run full comparison
    results = comparator.run()
    
    corr = results['correlation']
    overlap_128 = results['overlap'][128]
    
    print(f"\nCorrelation: {corr['pearson_r']:.4f} (Pearson), {corr['spearman_r']:.4f} (Spearman)")
    print(f"Top-128 Overlap: {overlap_128['overlap_count']}/128 ({overlap_128['overlap_percentage']:.2f}%)")
    print(f"Jaccard Similarity: {overlap_128['jaccard']:.4f}")
    
if __name__ == "__main__":
    main()