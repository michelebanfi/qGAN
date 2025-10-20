import numpy as np
import matplotlib.pyplot as plt


def plot_unified_comparison(target_probabilities, rbm_results, qgan_result, output_path='media/unified_comparison.png'):
    """
    Create a unified visualization comparing RBM and qGAN results.
    
    Args:
        target_probabilities (np.ndarray): Target distribution
        rbm_results (dict): Results from RBM training
        qgan_result (dict): Results from qGAN training
        output_path (str): Path to save the figure
    """
    # Create figure with subplots
    n_rbm = len(rbm_results)
    fig, axes = plt.subplots(2, n_rbm + 1, figsize=(20, 10))
    fig.suptitle('Generative Models Comparison: RBMs vs qGAN', fontsize=18, fontweight='bold')
    
    bar_width = 0.35
    index = np.arange(4)
    labels = ['00', '01', '10', '11']
    
    # First row: RBM results
    rbm_configs = sorted(rbm_results.keys())
    for i, n_hidden in enumerate(rbm_configs):
        ax = axes[0, i]
        P_model = rbm_results[n_hidden]['P_model']
        KL = rbm_results[n_hidden]['KL']
        
        ax.bar(index, target_probabilities, bar_width, label='Target', color='black', alpha=0.7)
        ax.bar(index + bar_width, P_model, bar_width, label='RBM', color='blue', alpha=0.7)
        
        ax.set_title(f'RBM (H={n_hidden})\nKL={KL:.6f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('State')
        if i == 0:
            ax.set_ylabel('Probability')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim([0, max(target_probabilities) * 1.2])
    
    # Last column of first row: qGAN result
    ax = axes[0, n_rbm]
    P_qgan = qgan_result['distribution']
    KL_qgan = qgan_result['KL']
    
    ax.bar(index, target_probabilities, bar_width, label='Target', color='black', alpha=0.7)
    ax.bar(index + bar_width, P_qgan, bar_width, label='qGAN', color='red', alpha=0.7)
    
    ax.set_title(f'qGAN\nKL={KL_qgan:.6f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('State')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim([0, max(target_probabilities) * 1.2])
    
    # Second row: Training metrics
    # RBM KL divergences comparison
    ax = axes[1, 0]
    rbm_kls = [rbm_results[h]['KL'] for h in rbm_configs]
    ax.bar(range(len(rbm_configs)), rbm_kls, color='blue', alpha=0.7)
    ax.set_title('RBM Final KL Divergences', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hidden Units')
    ax.set_ylabel('KL Divergence')
    ax.set_xticks(range(len(rbm_configs)))
    ax.set_xticklabels([f'H={h}' for h in rbm_configs])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # qGAN training curves
    if 'g_losses' in qgan_result:
        ax = axes[1, 1]
        ax.plot(qgan_result['g_losses'], label='Generator Loss', color='red', linewidth=2)
        ax.set_title('qGAN Generator Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.5)
    
    if 'kl_divergences' in qgan_result:
        ax = axes[1, 2]
        ax.plot(qgan_result['kl_divergences'], label='KL Divergence', color='green', linewidth=2)
        ax.set_title('qGAN KL Divergence Over Training', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.5)
    
    # Summary comparison
    ax = axes[1, 3]
    ax.axis('off')
    
    # Create summary text
    summary_text = "Summary:\n\n"
    summary_text += "RBM Results:\n"
    for h in rbm_configs:
        summary_text += f"  H={h}: KL={rbm_results[h]['KL']:.6f}, Params={rbm_results[h]['Params']}\n"
    summary_text += f"\nqGAN Result:\n"
    summary_text += f"  KL={KL_qgan:.6f}\n"
    summary_text += f"  Final G Loss={qgan_result.get('g_losses', [0])[-1]:.6f}\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Unified comparison saved to {output_path}")
    plt.close()


def plot_individual_qgan_metrics(qgan_result, output_dir='media'):
    """
    Plot individual qGAN training metrics.
    
    Args:
        qgan_result (dict): Results from qGAN training
        output_dir (str): Directory to save plots
    """
    if 'g_losses' in qgan_result:
        plt.figure(figsize=(8, 5))
        plt.plot(qgan_result['g_losses'], label='Generator Loss', linewidth=2)
        plt.title('qGAN Generator Loss During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(f'{output_dir}/qgan_generator_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if 'kl_divergences' in qgan_result:
        plt.figure(figsize=(8, 5))
        plt.plot(qgan_result['kl_divergences'], label='KL Divergence', color='green', linewidth=2)
        plt.title('qGAN KL Divergence During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(f'{output_dir}/qgan_kl_divergence.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_individual_rbm_results(target_probabilities, rbm_results, output_path='media/rbm_comparison.png'):
    """
    Plot RBM results separately.
    
    Args:
        target_probabilities (np.ndarray): Target distribution
        rbm_results (dict): Results from RBM training
        output_path (str): Path to save the figure
    """
    hidden_units_configs = sorted(rbm_results.keys())
    fig, axes = plt.subplots(1, len(hidden_units_configs), figsize=(15, 5), sharey=True)
    fig.suptitle('RBM vs. Target Distribution Comparison', fontsize=16)
    
    bar_width = 0.35
    index = np.arange(4)
    labels = ['00', '01', '10', '11']
    
    for i, n_hidden in enumerate(hidden_units_configs):
        ax = axes[i] if len(hidden_units_configs) > 1 else axes
        P_model = rbm_results[n_hidden]['P_model']
        KL = rbm_results[n_hidden]['KL']
        
        ax.bar(index, target_probabilities, bar_width, label='Target (P_data)', color='k', alpha=0.7)
        ax.bar(index + bar_width, P_model, bar_width, label='RBM (P_model)')
        
        ax.set_title(f'RBM H={n_hidden} (KL={KL:.6f})')
        ax.set_xlabel('State')
        if i == 0:
            ax.set_ylabel('Probability')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"RBM comparison saved to {output_path}")
    plt.close()
