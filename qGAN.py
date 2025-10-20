import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam
import torch.nn.functional as F

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN


class QuantumGAN:
    """
    A Quantum Generative Adversarial Network implementation.
    """
    
    def __init__(self, target_probabilities, learning_rate=0.01):
        """
        Initialize the quantum GAN.
        
        Args:
            target_probabilities (np.ndarray): Target distribution to learn
            learning_rate (float): Learning rate for optimizers
        """
        self.target_probabilities = target_probabilities
        self.real_data = torch.tensor(target_probabilities, dtype=torch.float32).reshape(1, 4)
        self.real_labels = torch.tensor([[1.]])
        self.fake_labels = torch.tensor([[0.]])
        
        # Build quantum generator
        self.q_generator = self._build_generator()
        
        # Build classical discriminator
        self.discriminator = self._build_discriminator()
        
        # Setup optimizers
        self.loss_func = BCELoss()
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=learning_rate)
        self.g_optimizer = Adam(list(self.q_generator.parameters()), lr=learning_rate)
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.kl_divergences = []
        
    def _build_generator(self):
        """Build the quantum generator circuit."""
        # Define two learnable parameters
        theta1 = Parameter('θ₁')
        theta2 = Parameter('θ₂')
        
        # Create a 2-qubit quantum circuit
        generator_circuit = QuantumCircuit(2)
        
        # Layer 1: Parameterized Rotations
        generator_circuit.ry(theta1, 0)
        generator_circuit.ry(theta2, 1)
        
        # Layer 2: Entanglement
        generator_circuit.cx(0, 1)
        
        # Create quantum neural network
        sampler = Sampler()
        qnn = SamplerQNN(
            circuit=generator_circuit,
            sampler=sampler,
            input_params=[], 
            weight_params=[theta1, theta2],
            sparse=False
        )
        
        return TorchConnector(qnn)
    
    def _build_discriminator(self):
        """Build the classical discriminator network."""
        return torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
    
    def train(self, num_epochs=100, verbose=True):
        """
        Train the quantum GAN.
        
        Args:
            num_epochs (int): Number of training epochs
            verbose (bool): Whether to print progress
        """
        for epoch in range(num_epochs):
            # 1. Train Discriminator
            self.d_optimizer.zero_grad()
            
            # On real data
            real_pred = self.discriminator(self.real_data)
            real_loss = self.loss_func(real_pred, self.real_labels)
            
            # On fake data
            fake_data = self.q_generator().reshape(1, 4)
            fake_pred = self.discriminator(fake_data.detach())
            fake_loss = self.loss_func(fake_pred, self.fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # 2. Train Generator
            self.g_optimizer.zero_grad()
            g_loss = self.loss_func(
                self.discriminator(self.q_generator().reshape(1, 4)), 
                self.real_labels
            )
            g_loss.backward()
            self.g_optimizer.step()
            
            # Record losses
            self.g_losses.append(g_loss.item())
            self.d_losses.append(d_loss.item())
            
            # Calculate KL divergence
            with torch.no_grad():
                generated_dist = self.q_generator().reshape(1, 4)
                kl_div = F.kl_div(generated_dist.log(), self.real_data, reduction='batchmean')
                self.kl_divergences.append(kl_div.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {g_loss.item():.6f}, "
                      f"D Loss: {d_loss.item():.6f}, KL Div: {kl_div.item():.6f}")
        
        if verbose:
            print("Training finished.")
    
    def get_generated_distribution(self):
        """
        Get the final generated distribution.
        
        Returns:
            np.ndarray: Generated probability distribution
        """
        return self.q_generator().detach().numpy().flatten()
