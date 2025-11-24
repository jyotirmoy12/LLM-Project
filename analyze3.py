
import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt


class ToxicVectorRoleAnalyzer:
   
    def __init__(self, 
                 probe_path='models/toxic_probe_final.pt',
                 counterspeech_path='models/counterspeech_generator_final.pt',
                 model_name='gpt2-medium'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        probe_checkpoint = torch.load(probe_path, map_location=self.device)
        self.w_toxic = probe_checkpoint['probe_weights'][1]

        cs_checkpoint = torch.load(counterspeech_path, map_location=self.device)
        
        # Initialize tokenizer with special tokens
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['[TOXIC]', '[RESPONSE]']}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model and resize embeddings
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load state dict
        cs_state_dict = cs_checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in cs_state_dict.items():
            if key.startswith('gpt2.'):
                new_key = key[5:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        self.config = self.model.config
        self.n_layers = self.config.n_layer
        self.hidden_size = self.config.n_embd
        
        print(f"Model loaded: {self.n_layers} layers, hidden_size={self.hidden_size}")
    
    def extract_mlp_vectors_with_activations(self, toxic_prompts):
    
        # Store activations for each layer's MLP
        layer_activations = {i: [] for i in range(self.n_layers)}
        
        # Generate counterspeech and capture activations
        with torch.no_grad():
            for prompt in tqdm(toxic_prompts, desc="Processing prompts"):
                formatted_prompt = f"[TOXIC] {prompt} [RESPONSE]"
                input_ids = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)
                
                # Forward pass with hooks to capture MLP activations
                hooks = []
                activations = {i: None for i in range(self.n_layers)}
                
                def get_activation_hook(layer_idx):
                    def hook(module, input, output):
                        # MLP output after c_proj
                        activations[layer_idx] = output.detach()
                    return hook
                
                # Register hooks
                for i in range(self.n_layers):
                    hook = self.model.transformer.h[i].mlp.register_forward_hook(
                        get_activation_hook(i)
                    )
                    hooks.append(hook)
                
                # Generate
                _ = self.model(input_ids)
                
                # Store activations
                for i in range(self.n_layers):
                    if activations[i] is not None:
                        # Average over sequence length
                        act = activations[i].mean(dim=1).squeeze().cpu().numpy()
                        layer_activations[i].append(act)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
        
        # Average activations across prompts
        avg_activations = {}
        for layer in range(self.n_layers):
            if layer_activations[layer]:
                avg_activations[layer] = np.mean(layer_activations[layer], axis=0)
        
        return avg_activations
    
    def classify_vectors_by_toxicity(self, threshold=0.1):

        print(f"\nClassifying vectors (threshold={threshold})")
        
        toxic_vectors = []
        non_toxic_vectors = []
        
        for layer in tqdm(range(self.n_layers), desc="Classifying"):
            mlp = self.model.transformer.h[layer].mlp
            w_proj = mlp.c_proj.weight.data.cpu().numpy()
            
            for neuron_idx in range(w_proj.shape[0]):
                value_vector = w_proj[neuron_idx, :]
                
                # Compute similarity with W_toxic
                sim = cosine_similarity(
                    self.w_toxic.reshape(1, -1),
                    value_vector.reshape(1, -1)
                )[0, 0]
                
                if sim > threshold:
                    toxic_vectors.append((layer, neuron_idx, sim))
                else:
                    non_toxic_vectors.append((layer, neuron_idx, sim))
        
        print(f"Found {len(toxic_vectors)} toxic vectors (>{threshold})")
        print(f"Found {len(non_toxic_vectors)} non-toxic vectors (<={threshold})")
        
        return toxic_vectors, non_toxic_vectors
    
    def analyze_activation_patterns(self, avg_activations, toxic_vectors, non_toxic_vectors):
        
        # Group vectors by layer
        toxic_by_layer = {}
        nontoxic_by_layer = {}
        
        for layer, neuron_idx, sim in toxic_vectors:
            if layer not in toxic_by_layer:
                toxic_by_layer[layer] = []
            toxic_by_layer[layer].append(neuron_idx)
        
        for layer, neuron_idx, sim in non_toxic_vectors:
            if layer not in nontoxic_by_layer:
                nontoxic_by_layer[layer] = []
            nontoxic_by_layer[layer].append(neuron_idx)
        
        # Compute mean activations for toxic vs non-toxic neurons
        results = {}
        
        for layer in range(self.n_layers):
            if layer not in avg_activations:
                continue
            
            acts = avg_activations[layer]
            
            # Get activations of toxic neurons
            toxic_acts = []
            if layer in toxic_by_layer:
                for neuron_idx in toxic_by_layer[layer]:
                    if neuron_idx < len(acts):
                        toxic_acts.append(acts[neuron_idx])
            
            # Get activations of non-toxic neurons
            nontoxic_acts = []
            if layer in nontoxic_by_layer:
                for neuron_idx in nontoxic_by_layer[layer]:
                    if neuron_idx < len(acts):
                        nontoxic_acts.append(acts[neuron_idx])
            
            results[layer] = {
                'toxic_mean': np.mean(toxic_acts) if toxic_acts else 0,
                'toxic_std': np.std(toxic_acts) if toxic_acts else 0,
                'nontoxic_mean': np.mean(nontoxic_acts) if nontoxic_acts else 0,
                'nontoxic_std': np.std(nontoxic_acts) if nontoxic_acts else 0,
                'toxic_count': len(toxic_acts),
                'nontoxic_count': len(nontoxic_acts)
            }
        
        return results
    
    def visualize_results(self, activation_results, toxic_vectors, non_toxic_vectors, 
                         output_dir='outputs'):
       
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Activation comparison
        self.plot_activation_comparison(activation_results, 
                                       f'{output_dir}/h2_activation_comparison.png')
    
    
    def plot_activation_comparison(self, results, save_path):
        
        layers = sorted(results.keys())
        toxic_means = [results[l]['toxic_mean'] for l in layers]
        nontoxic_means = [results[l]['nontoxic_mean'] for l in layers]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Mean activations
        ax1 = axes[0]
        ax1.plot(layers, toxic_means, marker='o', linewidth=2, 
                label='Toxic Vectors', color='#C5504B')
        ax1.plot(layers, nontoxic_means, marker='s', linewidth=2, 
                label='Non-Toxic Vectors', color='#70AD47')
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_ylabel('Mean Activation', fontsize=11)
        ax1.set_title('Activation During Counterspeech Generation', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Activation ratio
        ax2 = axes[1]
        ratios = []
        for l in layers:
            toxic_m = results[l]['toxic_mean']
            nontoxic_m = results[l]['nontoxic_mean']
            if nontoxic_m > 0:
                ratio = toxic_m / nontoxic_m
            else:
                ratio = 0
            ratios.append(ratio)
        
        colors = ['#C5504B' if r > 1 else '#70AD47' for r in ratios]
        ax2.bar(layers, ratios, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal activation')
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_ylabel('Toxic/Non-Toxic Activation Ratio', fontsize=11)
        ax2.set_title('Relative Contribution by Layer', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():

    # Initialize analyzer
    analyzer = ToxicVectorRoleAnalyzer(
        probe_path='models/toxic_probe_final.pt',
        counterspeech_path='models/counterspeech_generator_final.pt'
    )
    
    # Test prompts
    test_prompts = [
        "You're so stupid!",
        "I hate people like you.",
        "Get lost, nobody wants you here.",
        "All immigrants are criminals.",
        "Women don't belong in tech.",
        "You people are ruining everything.",
        "Go back to where you came from.",
        "You're worthless and pathetic."
    ]
    
    # Extract activations
    avg_activations = analyzer.extract_mlp_vectors_with_activations(test_prompts)
    
    # Classify vectors
    toxic_vectors, non_toxic_vectors = analyzer.classify_vectors_by_toxicity(threshold=0.1)
    
    # Analyze activation patterns
    activation_results = analyzer.analyze_activation_patterns(
        avg_activations, toxic_vectors, non_toxic_vectors
    )
    
    # Visualize
    analyzer.visualize_results(activation_results, toxic_vectors, non_toxic_vectors)
    
if __name__ == "__main__":
    main()