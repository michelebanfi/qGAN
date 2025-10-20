"""
Main training script for comparing RBM and qGAN models.

This script orchestrates the entire training and comparison process:
1. Generates target data distribution
2. Trains RBM models with different configurations
3. Trains quantum GAN model
4. Creates unified visualizations comparing both approaches
"""

import numpy as np
import torch
import torch.nn.functional as F
import os

from data_generator import generate_target_distribution
from RBM import RBMModel
from qGAN import QuantumGAN
from visualization import plot_unified_comparison


def main():
    """Main execution function."""
    
    print("=" * 60)
    print("Quantum GAN vs RBM Comparison")
    print("=" * 60)
    
    # Ensure media directory exists
    os.makedirs('media', exist_ok=True)
    
    # 1. Generate target distribution
    print("\n[1/4] Generating target distribution...")
    target_probabilities, bin_edges, samples = generate_target_distribution()
    print(f"Target Probabilities: {target_probabilities}")
    
    # 2. Train RBM models
    print("\n[2/4] Training RBM models...")
    print("-" * 60)
    rbm_model = RBMModel(
        target_probabilities=target_probabilities,
        hidden_units_configs=[1, 2, 3],
        n_train_samples=5000,
        learning_rate=0.01,
        n_iter=200
    )
    rbm_model.train(verbose=True)
    rbm_results = rbm_model.get_results()
    
    # 3. Train qGAN model
    print("\n[3/4] Training qGAN model...")
    print("-" * 60)
    qgan = QuantumGAN(
        target_probabilities=target_probabilities,
        learning_rate=0.01
    )
    qgan.train(num_epochs=200, verbose=True)
    
    # Get qGAN results
    qgan_distribution = qgan.get_generated_distribution()
    qgan_kl = F.kl_div(
        torch.tensor(qgan_distribution).log(),
        torch.tensor(target_probabilities),
        reduction='batchmean'
    ).item()
    
    qgan_result = {
        'distribution': qgan_distribution,
        'KL': qgan_kl,
        'g_losses': qgan.g_losses,
        'kl_divergences': qgan.kl_divergences
    }
    
    # 4. Generate visualizations
    print("\n[4/4] Generating visualizations...")
    print("-" * 60)
    
    # Unified comparison
    plot_unified_comparison(
        target_probabilities=target_probabilities,
        rbm_results=rbm_results,
        qgan_result=qgan_result,
        output_path='media/unified_comparison.png'
    )
    
    # 5. Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTarget Distribution: {target_probabilities}")
    print(f"\nRBM Results:")
    for h in sorted(rbm_results.keys()):
        result = rbm_results[h]
        print(f"  H={h}: KL={result['KL']:.6f}, Params={result['Params']}, Dist={result['P_model']}")
    
    print(f"\nqGAN Result:")
    print(f"  KL={qgan_kl:.6f}")
    print(f"  Distribution={qgan_distribution}")
    print(f"  Final Generator Loss={qgan.g_losses[-1]:.6f}")

    # Additional: print and save parameter matrices/tensors for comparison
    print("\nRBM learned parameters (per hidden size):")
    for h in sorted(rbm_results.keys()):
        res = rbm_results[h]
        print(f"\n  H={h}")
        print(f"    W shape: {res['W'].shape}\n{res['W']}")
        print(f"    visible_bias: {res['visible_bias']}")
        print(f"    hidden_bias: {res['hidden_bias']}")
        # Save to disk
        np.save(f"media/rbm_W_H{h}.npy", res['W'])
        np.save(f"media/rbm_visible_bias_H{h}.npy", res['visible_bias'])
        np.save(f"media/rbm_hidden_bias_H{h}.npy", res['hidden_bias'])

    # qGAN generator weights and unitary
    try:
        g_weights = qgan.get_generator_weights().numpy()
        U = qgan.get_generator_unitary()
        print("\nqGAN generator weights (θ₁, θ₂, σ₁, σ₂):", g_weights)
        print("qGAN generator unitary U (4x4 complex):\n", U)
        # Save to disk
        np.save("media/qgan_weights.npy", g_weights)
        np.save("media/qgan_unitary.npy", U)
    except Exception as e:
        print("Warning: could not extract qGAN unitary due to:", e)
    
    print("\n" + "=" * 60)
    print("All outputs saved to 'media/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()