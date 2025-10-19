import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import BCELoss
from torch.optim import Adam
import torch.nn.functional as F

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

num_samples = 1000
mu, sigma = 1, 0.5  # Parameters for the log-normal distribution
samples = np.random.lognormal(mu, sigma, num_samples)

mean = np.mean(samples)
std = np.std(samples)
min_val = np.min(samples)
max_val = np.max(samples)

bin_edges = [min_val, mean - 0.5*std, mean, mean + 0.5*std, max_val]

num_bins = 4
counts, _ = np.histogram(samples, bins=bin_edges)

probabilities = counts / np.sum(counts)

real_data = torch.tensor(probabilities, dtype=torch.float32).reshape(1, 4)

real_labels = torch.tensor([[1.]])
fake_labels = torch.tensor([[0.]])

num_epochs = 100

# Define two learnable parameters (our 'knobs')
theta1 = Parameter('θ₁')
theta2 = Parameter('θ₂')

# Create a 2-qubit quantum circuit
generator_circuit = QuantumCircuit(2)

# --- Layer 1: Parameterized Rotations ---
generator_circuit.ry(theta1, 0) # Apply Ry(θ₁) to qubit 0
generator_circuit.ry(theta2, 1) # Apply Ry(θ₂) to qubit 1

# --- Layer 2: Entanglement ---
generator_circuit.cx(0, 1)      # Apply CNOT from qubit 0 to 1

# Let's see the circuit
print(generator_circuit.draw(output="text"))

sampler = Sampler()

qnn = SamplerQNN(
    circuit=generator_circuit,
    sampler=sampler,
    input_params=[], 
    weight_params=[theta1, theta2],
    sparse=False
)

q_generator = TorchConnector(qnn)

discriminator = torch.nn.Sequential(
    torch.nn.Linear(4, 256), # Input size 4 for our 2-qubit system
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)

loss_func = BCELoss()
d_optimizer = Adam(discriminator.parameters(), lr=0.01)
g_optimizer = Adam(list(q_generator.parameters()), lr=0.01)

g_losses = []
kl_divergences = []

for epoch in range(num_epochs):
    # 1. Train Discriminator
    d_optimizer.zero_grad()
    
    # On real data (assuming we have 'real_data' and 'real_labels' = 1)
    real_pred = discriminator(real_data)
    real_loss = loss_func(real_pred, real_labels)
    
    # On fake data (labels = 0)
    fake_data = q_generator().reshape(1, 4) # Get probabilities from the quantum circuit
    fake_pred = discriminator(fake_data.detach())
    fake_loss = loss_func(fake_pred, fake_labels)
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    
    # 2. Train Generator
    g_optimizer.zero_grad()
    
    # We want the discriminator to think the fake data is real (label = 1)
    g_loss = loss_func(discriminator(q_generator().reshape(1, 4)), real_labels)
    
    g_loss.backward()
    g_optimizer.step()
    
    g_losses.append(g_loss.item())
    
    with torch.no_grad():
        generated_dist = q_generator().reshape(1, 4)
        # The kl_div function expects log-probabilities for the first input
        kl_div = F.kl_div(generated_dist.log(), real_data, reduction='batchmean')
        kl_divergences.append(kl_div.item())
    

print("Training finished.")

# --- 6. Check Results ---
final_fake_data = q_generator().detach().numpy().flatten()
print("\nTarget Distribution:", real_data.numpy().flatten())
print("Generated Distribution:", final_fake_data)

# --- 7. Plotting ---
plt.figure(figsize=(8, 5))
plt.plot(g_losses, label='Generator Loss')
plt.title('Generator Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("generator_loss.png")

plt.figure(figsize=(8, 5))
plt.plot(kl_divergences, label='KL Divergence')
plt.title('KL Divergence During Training')
plt.xlabel('Epoch')
plt.ylabel('Divergence')
plt.legend()
plt.grid(True)
plt.savefig("kl_divergence.png")

plt.figure(figsize=(8, 5))
bar_width = 0.35
index = np.arange(4)
plt.bar(index, real_data.numpy().flatten(), bar_width, label='Real Data')
plt.bar(index + bar_width, final_fake_data, bar_width, label='Generated Data')
plt.title('Comparison of Real and Generated Distributions')
plt.xlabel('State')
plt.ylabel('Probability')
plt.xticks(index + bar_width / 2, ['00', '01', '10', '11'])
plt.legend()
plt.grid(True)
plt.savefig("generated_vs_real.png")