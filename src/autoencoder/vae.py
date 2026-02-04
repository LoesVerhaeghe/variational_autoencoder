import torch.nn as nn
import torch
from src.autoencoder.model_structure import Autoencoder

class VAE(Autoencoder):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Add mu and log_var layers for reparameterization
        self.latent_channels = 8
        self.spatial_dims = (6, 8)  # adjust if needed based on input size
        
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        self.to(self.device)

    def reparameterize(self, mu, log_var): # reparameterization trick to make sampling from the latent distribution differentiable
        std = torch.exp(0.5 * log_var) # Convert log-variance to std deviation
        eps = torch.randn_like(std) # Sample noise from standard normal
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)

        mu = self.mu(encoded)
        log_var = self.log_var(encoded)

        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(z)
        
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.num_hidden).to(self.device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples