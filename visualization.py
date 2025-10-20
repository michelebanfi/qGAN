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
    # Create a single figure with grouped bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    states = ['00', '01', '10', '11']
    x = np.arange(len(states))
    
    # Get RBM results sorted by hidden units
    rbm_configs = sorted(rbm_results.keys())
    n_models = len(rbm_configs) + 2  # RBMs + qGAN + Target
    
    bar_width = 0.15
    
    # Plot target distribution
    ax.bar(x - bar_width * 2, target_probabilities, bar_width, 
           label='Target Distribution', color='blue', alpha=0.8)
    
    # Plot RBM models
    colors = ['orange', 'green', 'red']
    for i, n_hidden in enumerate(rbm_configs):
        P_model = rbm_results[n_hidden]['P_model']
        KL = rbm_results[n_hidden]['KL']
        offset = -bar_width * 2 + bar_width * (i + 1)
        ax.bar(x + offset, P_model, bar_width, 
               label=f'RBM H={n_hidden} (KL={KL:.6f})', 
               color=colors[i], alpha=0.8)
    
    # Plot qGAN
    P_qgan = qgan_result['distribution']
    KL_qgan = qgan_result['KL']
    offset = -bar_width * 2 + bar_width * (len(rbm_configs) + 1)
    ax.bar(x + offset, P_qgan, bar_width, 
           label=f'qGAN (KL={KL_qgan:.6f})', 
           color='purple', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('State (Binary)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Target and Generated Distributions (RBMs vs qGAN)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim([0, max(target_probabilities) * 1.15])
    
    plt.tight_layout()
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
