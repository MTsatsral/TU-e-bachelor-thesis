import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import rlcodebase
# the type of agent is randomly choosen, haven't looked into which one works the best in our case
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import Config, Logger
from torch.utils.tensorboard import SummaryWriter
from model import Policy

parser = argparse.ArgumentParser()
parser.add_argument('--use-encoder', default=False, action='store_true')
parser.add_argument('--encoder-path', default='./', type=str)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--port', default=2000, type=int)
parser.add_argument('--latent-size', default=16, type=int, help='dimension of latent state embedding')
parser.add_argument('--model-save-path', default='./checkpoints/policy.pt', type=str)
args = parser.parse_args()

# General Question: where is the optimal place to implement the part for 'faking' the uncertain domain labels to the data?

# Question: is this encoder the "backbone" we want?
class Encoder(nn.Module):
    def __init__(self, latent_size = 16, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(1024, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu


class some_function_for_rl_env(..., encoder):
    # TODO: fill in this function
    # Questions:
    # this function is often used to generate the 'environment' in the agent, what does envirment define?
    # what should we possibly include in this function?



def main():
    # prepare config
    # TODO: will investigate what these params mean
    config = Config()
    config.algo = 'ppo'
    config.num_envs = 1
    config.optimizer = 'Adam'
    config.lr = args.lr
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.use_grad_clip = True
    config.max_grad_norm = 0.5
    config.rollout_length = 128
    config.value_loss_coef = 1
    config.entropy_coef = 0.01
    config.ppo_epoch = 3
    config.ppo_clip_param = 0.2
    config.num_mini_batch = 3
    config.use_gpu = True
    config.save_interval = 10000
    config.memory_on_gpu = True
    config.after_set()
    print(config)


    encoder = Encoder(latent_size = args.latent_size)
    # load the common representation optimized by VAE
    weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(weights)

    # prepare environment
    env = some_function_for_rl_env(..., encoder)

    # prepare model
    input_dim = args.latent_size+1
    model = Policy(input_dim, 2).to(config.device) # import from model.py

    # create agent and run
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)
    agent = PPOAgent(config, env, model, logger)
    agent.run()
    torch.save(model.state_dict(), args.model_save_path)

if __name__ == '__main__':
    main()