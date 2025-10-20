import numpy as np

# Set a seed for reproducibility
np.random.seed(42)


def generate_target_distribution():
    """
    Generates the target distribution that both RBM and qGAN should learn.
    
    Returns:
        target_probabilities (np.ndarray): Probabilities for each state [00, 01, 10, 11]
        bin_edges (list): Edges used for binning the samples
        samples (np.ndarray): Raw log-normal samples
    """
    num_samples = 1000
    mu, sigma = 1, 0.5
    samples = np.random.lognormal(mu, sigma, num_samples)
    
    mean = np.mean(samples)
    std = np.std(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    # Define the bin edges
    bin_edges = [min_val, mean - 0.5*std, mean, mean + 0.5*std, max_val]
    
    # Calculate the target probabilities
    counts, _ = np.histogram(samples, bins=bin_edges)
    target_probabilities = counts / np.sum(counts)
    
    return target_probabilities, bin_edges, samples


def generate_rbm_training_data(target_probabilities, n_samples=5000):
    """
    Generates training samples for the RBM based on target probabilities.
    
    Args:
        target_probabilities (np.ndarray): Target distribution probabilities
        n_samples (int): Number of training samples to generate
        
    Returns:
        X_train (np.ndarray): Training dataset of shape (n_samples, 2)
    """
    # The 4 possible states (00, 01, 10, 11)
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Generate indices based on the target probabilities
    sample_indices = np.random.choice(4, size=n_samples, p=target_probabilities)
    # The training dataset
    X_train = states[sample_indices]
    
    return X_train
