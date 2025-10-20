import numpy as np
from sklearn.neural_network import BernoulliRBM
import itertools
import warnings

# Set a seed for reproducibility
np.random.seed(42)


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


class RBMModel:
    """
    Restricted Boltzmann Machine model wrapper.
    """
    
    def __init__(self, target_probabilities, hidden_units_configs=[1, 2, 3], 
                 n_train_samples=5000, learning_rate=0.01, n_iter=200):
        """
        Initialize RBM model.
        
        Args:
            target_probabilities (np.ndarray): Target distribution to learn
            hidden_units_configs (list): List of hidden unit configurations to test
            n_train_samples (int): Number of training samples
            learning_rate (float): Learning rate for RBM
            n_iter (int): Number of training iterations
        """
        self.target_probabilities = target_probabilities
        self.hidden_units_configs = hidden_units_configs
        self.n_train_samples = n_train_samples
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.N_V = 2  # Number of visible units
        self.results = {}
        
    def _prepare_training_data(self):
        """Prepare training data for RBM."""
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        sample_indices = np.random.choice(4, size=self.n_train_samples, p=self.target_probabilities)
        return states[sample_indices]
    
    def train(self, verbose=True):
        """
        Train RBM models with different hidden unit configurations.
        
        Args:
            verbose (bool): Whether to print progress
        """
        X_train = self._prepare_training_data()
        
        # Suppress potential convergence warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for n_hidden in self.hidden_units_configs:
            if verbose:
                print(f"Training RBM with H={n_hidden}...")
            
            # Initialize and train RBM
            rbm = BernoulliRBM(
                n_components=n_hidden,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                verbose=0,
                random_state=42
            )
            rbm.fit(X_train)
            
            # Evaluate the RBM
            P_model = calculate_exact_probabilities(rbm, self.N_V)
            kl_div = kl_divergence_numpy(self.target_probabilities, P_model)
            n_params = (self.N_V * n_hidden) + self.N_V + n_hidden
            
            self.results[n_hidden] = {
                'P_model': P_model,
                'KL': kl_div,
                'Params': n_params
            }
            
            if verbose:
                print(f"  Parameters: {n_params}")
                print(f"  Learned Probabilities: {P_model}")
                print(f"  KL Divergence: {kl_div:.6f}\n")
        
        warnings.filterwarnings("default", category=UserWarning)
    
    def get_results(self):
        """
        Get training results.
        
        Returns:
            dict: Dictionary containing results for each configuration
        """
        return self.results