import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import itertools
import warnings

# Set a seed for reproducibility
np.random.seed(42)

# === 1. Helper Functions ===

def kl_divergence_numpy(P_data, P_model):
    """
    Calculates D_KL(P_data || P_model).
    P_data: Target distribution (probabilities).
    P_model: Learned distribution (probabilities).
    """
    # Epsilon to avoid log(0)
    epsilon = 1e-15
    # Ensure distributions are clipped safely
    P_model_safe = np.clip(P_model, epsilon, 1.0)
    P_data_safe = np.clip(P_data, epsilon, 1.0)
    
    # D_KL(P || Q) = sum [ P(x) * log(P(x) / Q(x)) ]
    kl_div = np.sum(P_data_safe * np.log(P_data_safe / P_model_safe))
    return kl_div

def calculate_exact_probabilities(rbm, N_V):
    """Calculates the exact P(v) learned by the RBM by calculating the partition function Z."""
    W = rbm.components_.T # (n_visible, n_hidden)
    v_bias = rbm.intercept_visible_
    h_bias = rbm.intercept_hidden_
    
    # Generate all possible visible configurations
    visible_configs = np.array(list(itertools.product([0, 1], repeat=N_V)))
    
    # Calculate the unnormalized log probability (proportional to -Free Energy)
    # log P_unnorm(v) = v·b + sum_j log(1 + exp(v·W_j + c_j))

    # Term 1: v·b
    visible_bias_term = visible_configs @ v_bias

    # Term 2: Hidden unit contribution
    hidden_activation = visible_configs @ W + h_bias
    # Use np.logaddexp(0, x) which is equivalent to log(1 + exp(x)) for stability
    softplus_term = np.sum(np.logaddexp(0, hidden_activation), axis=1)
    
    unnormalized_log_probs = visible_bias_term + softplus_term
        
    # Calculate the partition function Z using the log-sum-exp trick for numerical stability
    # log_Z = log(sum(exp(log_P_unnorm)))
    max_log = np.max(unnormalized_log_probs)
    log_Z = max_log + np.log(np.sum(np.exp(unnormalized_log_probs - max_log)))
    
    # Calculate P(v) = exp(log_P_unnorm) / Z
    log_P_v = unnormalized_log_probs - log_Z
    P_v = np.exp(log_P_v)
    
    return P_v

# === 2. Data Generation (Replicating the Qiskit setup) ===
# This defines the target distribution the models should learn.
num_samples = 1000
mu, sigma = 1, 0.5
samples = np.random.lognormal(mu, sigma, num_samples)

mean = np.mean(samples)
std = np.std(samples)
min_val = np.min(samples)
max_val = np.max(samples)

# Define the bin edges as specified in the Qiskit script
bin_edges = [min_val, mean - 0.5*std, mean, mean + 0.5*std, max_val]

# Calculate the target probabilities
counts, _ = np.histogram(samples, bins=bin_edges)
target_probabilities = counts / np.sum(counts)
print(f"Target Probabilities (P_data): {target_probabilities}\n")

# === 3. RBM Training Data Preparation ===
# RBMs train on samples (e.g., [0,0], [0,1]), not probabilities.
N_V = 2 # Number of visible units
N_RBM_TRAIN = 5000
# The 4 possible states (00, 01, 10, 11)
states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Generate indices based on the target probabilities
sample_indices = np.random.choice(4, size=N_RBM_TRAIN, p=target_probabilities)
# The training dataset
X_train = states[sample_indices]

# === 4. Training and Evaluation Loop ===

hidden_units_configs = [1, 2, 3]
results = {}

# Suppress potential convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning) 

for n_hidden in hidden_units_configs:
    print(f"Training RBM with H={n_hidden}...")
    
    # Initialize RBM
    # Hyperparameters found to work well: LR=0.01, default batch_size=10.
    rbm = BernoulliRBM(n_components=n_hidden, 
                       learning_rate=0.01, 
                       # batch_size=10, # Default value
                       n_iter=200, # Number of epochs (iterations)
                       verbose=0, 
                       random_state=42)
    
    # Train the RBM using Contrastive Divergence
    rbm.fit(X_train)
    
    # Evaluate the RBM exactly
    P_model = calculate_exact_probabilities(rbm, N_V)
    
    # Calculate KL Divergence D_KL(P_data || P_model)
    kl_div = kl_divergence_numpy(target_probabilities, P_model)
    
    # Calculate parameters
    n_params = (N_V * n_hidden) + N_V + n_hidden
    
    results[n_hidden] = {'P_model': P_model, 'KL': kl_div, 'Params': n_params}
    print(f"  Parameters: {n_params}")
    print(f"  Learned Probabilities: {P_model}")
    print(f"  KL Divergence: {kl_div:.6f}\n")

warnings.filterwarnings("default", category=UserWarning)

# === 5. Visualization ===

fig, axes = plt.subplots(1, len(hidden_units_configs), figsize=(15, 5), sharey=True)
fig.suptitle('RBM vs. Target Distribution Comparison', fontsize=16)

bar_width = 0.35
index = np.arange(4)
labels = ['00', '01', '10', '11']

for i, n_hidden in enumerate(hidden_units_configs):
    ax = axes[i]
    P_model = results[n_hidden]['P_model']
    KL = results[n_hidden]['KL']
    
    ax.bar(index, target_probabilities, bar_width, label='Target (P_data)', color='k', alpha=0.7)
    ax.bar(index + bar_width, P_model, bar_width, label=f'RBM (P_model)')
    
    ax.set_title(f'RBM H={n_hidden} (KL={KL:.6f})')
    ax.set_xlabel('State')
    if i == 0:
        ax.set_ylabel('Probability')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('rBM_vs_target_comparison.png')