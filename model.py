import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
from utils import reparameterize

# For VAE part
class Encoder(nn.Module):
    # Question: whether the choises for these values make sense?
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3, flatten_size=1024):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x


# DisentangledVAE here is actually Cycle-Consistent VAE
# disentangled stands for the disentanglement between domain-general and domain-specifc embeddings
class DisentangledVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, img_channel = 3, flatten_size = 1024):
        super(DisentangledVAE, self).__init__()
        self.encoder = Encoder(class_latent_size, content_latent_size, img_channel, flatten_size)
        self.decoder = Decoder(class_latent_size + content_latent_size, img_channel, flatten_size)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x

# reinforcement learning policy
class Policy(nn.Module):
    # this part is copied from a tutorial, which is quite different from our goal
    # TODO: possibly not universial, needs to be changed
    # Question: what should be inplemented here?
    def __init__(self, input_dim, action_dim, hidden_layer=[64, 64]):
        # Question: is this part universal? should we use it as it is?
        super(Policy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size) - 1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i + 1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())

        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size) - 1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i + 1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        # TODO: what to implement here?
        return None
