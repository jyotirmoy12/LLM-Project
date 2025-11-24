
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ToxicVectorVisualizer:
    
    def __init__(self, model_name='gpt2-medium', probe_model_path='toxic_probe_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.gpt2.eval()
        
        self.config = self.gpt2.config
        self.n_layers = self.config.n_layer
        self.hidden_size = self.config.n_embd
        self.mlp_size = self.config.n_inner
        
        # Load toxic probe weights
        checkpoint = torch.load(probe_model_path, map_location=self.device)
        self.w_toxic = checkpoint['probe_weights'][1]
        self.w_toxic_tensor = torch.tensor(self.w_toxic, device=self.device)
        
        print(f"Model: {self.n_layers} layers, hidden_size={self.hidden_size}")
    
    def extract_mlp_value_vectors(self):
        value_vectors = {}
        
        for layer_idx in tqdm(range(self.n_layers), desc="Extracting layers"):
            mlp = self.gpt2.transformer.h[layer_idx].mlp
            w_proj = mlp.c_proj.weight.data.cpu().numpy()
            
            for i in range(w_proj.shape[0]):
                value_vector = w_proj[i, :]
                value_vectors[(layer_idx, i)] = value_vector
        
        print(f"Extracted {len(value_vectors)} value vectors")
        return value_vectors
    
    def compute_cosine_similarities(self, value_vectors):
        similarities = {}
        
        for (layer, idx), v in tqdm(value_vectors.items(), desc="Computing"):
            sim = cosine_similarity(
                self.w_toxic.reshape(1, -1),
                v.reshape(1, -1)
            )[0, 0]
            similarities[(layer, idx)] = sim
        
        return similarities
    
    
    def create_layer_summary_plot(self, similarities, save_path='layer_summary.png'):
    
        # Group by layer
        layer_stats = {}
        for (layer, idx), sim in similarities.items():
            if layer not in layer_stats:
                layer_stats[layer] = []
            layer_stats[layer].append(sim)
        
        layers = sorted(layer_stats.keys())
        means = [np.mean(layer_stats[l]) for l in layers]
        maxs = [np.max(layer_stats[l]) for l in layers]
        mins = [np.min(layer_stats[l]) for l in layers]
        stds = [np.std(layer_stats[l]) for l in layers]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Mean similarity per layer
        ax1 = axes[0, 0]
        ax1.plot(layers, means, marker='o', linewidth=2, markersize=6, color='#4472C4')
        ax1.fill_between(layers, 
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          alpha=0.3, color='#4472C4')
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_ylabel('Mean Cosine Similarity', fontsize=11)
        ax1.set_title('Average Toxic Vector Similarity Across Layers', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Plot 2: Max similarity per layer
        ax2 = axes[0, 1]
        ax2.plot(layers, maxs, marker='s', linewidth=2, markersize=6, color='#ED7D31')
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_ylabel('Max Cosine Similarity', fontsize=11)
        ax2.set_title('Maximum Toxic Vector Similarity Across Layers', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution range per layer
        ax3 = axes[1, 0]
        ax3.fill_between(layers, mins, maxs, alpha=0.4, color='#70AD47')
        ax3.plot(layers, means, linewidth=2, color='#70AD47', label='Mean')
        ax3.set_xlabel('Layer', fontsize=11)
        ax3.set_ylabel('Cosine Similarity', fontsize=11)
        ax3.set_title('Similarity Range Across Layers', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Plot 4: Heatmap of similarity distribution
        ax4 = axes[1, 1]
        
        # Create binned heatmap
        n_bins = 30
        heatmap_data = np.zeros((len(layers), n_bins))
        bin_edges = np.linspace(-0.3, 0.3, n_bins + 1)
        
        for i, layer in enumerate(layers):
            sims = layer_stats[layer]
            counts, _ = np.histogram(sims, bins=bin_edges)
            heatmap_data[i, :] = counts / len(sims)  # Normalize
        
        im = ax4.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd', origin='lower')
        ax4.set_xlabel('Layer', fontsize=11)
        ax4.set_ylabel('Cosine Similarity Bin', fontsize=11)
        ax4.set_title('Similarity Distribution Heatmap', fontsize=12)
        ax4.set_xticks(range(0, len(layers), max(1, len(layers)//10)))
        ax4.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//10))])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Proportion', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer summary plot saved to {save_path}")
        plt.close()
    

def main():
    # Initialize visualizer
    visualizer = ToxicVectorVisualizer(
        model_name='gpt2-medium',
        probe_model_path='models/toxic_probe_final.pt'
    )
    
    # Extract value vectors
    value_vectors = visualizer.extract_mlp_value_vectors()
    
    # Compute similarities
    similarities = visualizer.compute_cosine_similarities(value_vectors)


if __name__ == "__main__":
    main()