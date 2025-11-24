"""
Enhanced visualization for toxic vector extraction
Creates plots similar to Figure 5 from the mechanistic understanding paper
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedToxicVectorVisualizer:
    """
    Enhanced visualizer with paper-style plots
    """
    
    def __init__(self, model_name='gpt2-medium', probe_model_path='toxic_probe_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load GPT-2 model
        print("Loading GPT-2 model...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.gpt2.eval()
        
        self.config = self.gpt2.config
        self.n_layers = self.config.n_layer
        self.hidden_size = self.config.n_embd
        self.mlp_size = self.config.n_inner
        
        # Load toxic probe weights
        print("Loading toxic probe weights...")
        checkpoint = torch.load(probe_model_path, map_location=self.device)
        self.w_toxic = checkpoint['probe_weights'][1]
        self.w_toxic_tensor = torch.tensor(self.w_toxic, device=self.device)
        
        print(f"Model: {self.n_layers} layers, hidden_size={self.hidden_size}")
    
    def extract_mlp_value_vectors(self):
        """Extract all MLP value vectors"""
        print("\nExtracting MLP value vectors...")
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
        """Compute cosine similarity with W_Toxic"""
        print("\nComputing cosine similarities...")
        similarities = {}
        
        for (layer, idx), v in tqdm(value_vectors.items(), desc="Computing"):
            sim = cosine_similarity(
                self.w_toxic.reshape(1, -1),
                v.reshape(1, -1)
            )[0, 0]
            similarities[(layer, idx)] = sim
        
        return similarities
    
    def compute_mean_activations(self, value_vectors, toxic_prompt_activations=None):
        """
        Compute mean activations for visualization
        If toxic_prompt_activations not provided, returns zeros as placeholder
        """
        if toxic_prompt_activations is None:
            # Placeholder: return zeros for demonstration
            mean_acts = {}
            for (layer, idx) in value_vectors.keys():
                mean_acts[(layer, idx)] = 0.0
            return mean_acts
        
        return toxic_prompt_activations
    
    def create_paper_style_plots(self, similarities, mean_activations=None, 
                                 save_path='toxic_vectors_paper_style.png'):
        """
        Create plots similar to Figure 5 from the paper
        Shows cosine similarity and mean activation distributions across layers
        """
        # Select subset of layers to visualize (similar to paper)
        layers_to_plot = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        if self.n_layers == 12:  # gpt2-medium has 24 layers, gpt2 has 12
            layers_to_plot = [0, 2, 4, 6, 8, 10]
        
        # Prepare data for each layer
        layer_data = {}
        for layer in layers_to_plot:
            cos_sims = []
            mean_acts = []
            
            for (l, idx), sim in similarities.items():
                if l == layer:
                    cos_sims.append(sim)
                    if mean_activations:
                        mean_acts.append(mean_activations.get((l, idx), 0.0))
                    else:
                        mean_acts.append(0.0)
            
            layer_data[layer] = {
                'cos_sim': np.array(cos_sims),
                'mean_act': np.array(mean_acts)
            }
        
        # Create figure similar to paper
        n_rows = 2
        n_cols = len(layers_to_plot) // n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6))
        fig.suptitle('Cosine Similarity between $\delta_T$ and $\delta_v^{l,h}$', 
                     fontsize=14, y=1.02)
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        for idx, layer in enumerate(layers_to_plot):
            ax = axes[idx]
            data = layer_data[layer]
            
            # Create histogram bins
            bins = np.linspace(-0.4, 0.4, 50)
            
            # Plot cosine similarity distribution
            counts_cos, edges_cos = np.histogram(data['cos_sim'], bins=bins)
            counts_cos = counts_cos / len(data['cos_sim'])  # Normalize to proportion
            
            # Blue bars for cosine similarity
            ax.bar(edges_cos[:-1], counts_cos, width=np.diff(edges_cos), 
                   color='#4472C4', alpha=0.7, edgecolor='none', label='Cos Sim')
            
            # If we have activation data, overlay it
            if mean_activations and np.any(data['mean_act'] != 0):
                counts_act, edges_act = np.histogram(data['mean_act'], bins=bins)
                counts_act = counts_act / len(data['mean_act'])
                ax.bar(edges_act[:-1], counts_act, width=np.diff(edges_act),
                       color='#ED7D31', alpha=0.6, edgecolor='none', label='Mean Act.')
            
            # Formatting
            ax.set_title(f'Layer {layer}', fontsize=11)
            ax.set_ylim(0, 0.24)
            ax.set_xlim(-0.3, 0.3)
            
            # Only show y-label on leftmost plots
            if idx % n_cols == 0:
                ax.set_ylabel('Proportion', fontsize=10)
            
            # Only show x-label on bottom plots
            if idx >= n_cols:
                ax.set_xlabel('Cos Sim', fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Add mean activation text below (like in paper)
            mean_cos = np.mean(data['cos_sim'])
            mean_act_val = np.mean(data['mean_act'])
            ax.text(0.5, -0.35, f'{mean_cos:.1f}\nMean Act.', 
                   transform=ax.transData, ha='center', va='top',
                   fontsize=8, color='#ED7D31')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPaper-style plot saved to {save_path}")
        plt.close()
    
    def create_layer_summary_plot(self, similarities, save_path='layer_summary.png'):
        """
        Create a summary plot showing statistics across all layers
        """
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
    
    def create_top_vectors_analysis(self, value_vectors, similarities, 
                                    top_k=128, save_path='top_vectors_analysis.png'):
        """
        Analyze and visualize the top toxic vectors
        """
        # Get top vectors
        sorted_vectors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_vectors = sorted_vectors[:top_k]
        
        # Analyze layer distribution
        layer_counts = {}
        for (layer, idx), sim in top_vectors:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Plot 1: Layer distribution of top toxic vectors
        ax1 = axes[0]
        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]
        ax1.bar(layers, counts, color='#C5504B', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_ylabel('Number of Top Toxic Vectors', fontsize=11)
        ax1.set_title(f'Distribution of Top-{top_k} Toxic Vectors Across Layers', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Similarity scores of top vectors
        ax2 = axes[1]
        similarities_values = [sim for (layer, idx), sim in top_vectors]
        ax2.plot(range(top_k), similarities_values, linewidth=2, color='#4472C4')
        ax2.set_xlabel(f'Rank (Top {top_k})', fontsize=11)
        ax2.set_ylabel('Cosine Similarity', fontsize=11)
        ax2.set_title('Similarity Scores of Top Toxic Vectors', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative layer contribution
        ax3 = axes[2]
        sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
        cumulative = np.cumsum([count for _, count in sorted_layers])
        layer_labels = [f"L{layer}" for layer, _ in sorted_layers]
        ax3.plot(range(len(cumulative)), cumulative, marker='o', linewidth=2, 
                markersize=6, color='#70AD47')
        ax3.axhline(y=top_k * 0.8, color='red', linestyle='--', 
                   linewidth=1, label='80% threshold')
        ax3.set_xlabel('Layer (sorted by contribution)', fontsize=11)
        ax3.set_ylabel('Cumulative Count', fontsize=11)
        ax3.set_title('Cumulative Layer Contribution', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xticks(range(len(cumulative)))
        ax3.set_xticklabels(layer_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top vectors analysis saved to {save_path}")
        plt.close()


def main():
    # Initialize visualizer
    visualizer = EnhancedToxicVectorVisualizer(
        model_name='gpt2-medium',
        probe_model_path='models/toxic_probe_final.pt'
    )
    
    # Extract value vectors
    value_vectors = visualizer.extract_mlp_value_vectors()
    
    # Compute similarities
    similarities = visualizer.compute_cosine_similarities(value_vectors)
    
    # Create paper-style plots
    visualizer.create_paper_style_plots(
        similarities, 
        save_path='toxic_vectors_paper_style.png'
    )
    
    # Create layer summary
    visualizer.create_layer_summary_plot(
        similarities,
        save_path='toxic_vectors_layer_summary.png'
    )
    
    # Create top vectors analysis
    visualizer.create_top_vectors_analysis(
        value_vectors,
        similarities,
        top_k=128,
        save_path='toxic_vectors_top_analysis.png'
    )
    
    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()