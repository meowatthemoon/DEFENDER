"""
Adapted from Phil Tabor's (Machine Learning with Phil) SAC implementation.
https://github.com/philtabor
"""

import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from models.memory import ReplayMemory


"""
SAC 
- Max Entropy framework.
- Scale the cost function that encourages exploration.
- SAC tends to be more smooth than other RL algorithms.
"""

class ActorNetwork(nn.Module):
    def __init__(self, lr :float, state_size : int, action_size : int, action_range, fc1_dims : int = 256, fc2_dims : int = 256):
        super(ActorNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.state_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_size) # mean
        self.sigma = nn.Linear(self.fc2_dims, self.action_size) # std

        self.optimizer = Adam(self.parameters(), lr = lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)

        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # STD determines width of the distribution, don't want it to be too broad
        sigma = torch.clamp(sigma, min = self.reparam_noise, max = 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize = True):
        """
        Returns actions in the range of the environment [-action_range, action_range], reparameterize means to explore
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # Sample distribution
        if reparameterize:
            # Sample + some noise (exploration)
            actions = probabilities.rsample() # sample for exploration
        else:
            # Sample without any noise (exploitation)
            actions = probabilities.sample()# sample 'best' action

        # Activate the action -1,1, multiply by the scale of the environment
        action = torch.tanh(actions) * torch.tensor(self.action_range).to(self.device) # multiply by action range

        # Get the log probs of the sampled actions from the distribution
        log_probs = probabilities.log_prob(actions) # Only for the loss function
        log_probs -= torch.log(1-action.pow(2) + self.reparam_noise) # noise to avoid log of 0
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
class CriticNetwork(nn.Module):
    def __init__(self, lr: float, state_size : int, action_size : int, fc1_dims :int = 256, fc2_dims : int = 256):
        super(CriticNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.state_size + self.action_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = Adam(self.parameters(), lr = lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim = 1))
        action_value = F.relu(action_value)

        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        return q

class ValueNetwork(nn.Module):
    def __init__(self, lr : float, state_size : int, fc1_dims : int = 256, fc2_dims : int = 256):
        super(ValueNetwork, self).__init__()

        self.state_size = state_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.state_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = Adam(self.parameters(), lr = lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

class SACAgent:
    def __init__(self, state_size : int, action_size : int, action_range, alpha : float = 0.0003, beta : float = 0.0003, gamma : float = 0.99, mem_size : int = 1000000, fc1_dims : int = 256, fc2_dims : int = 256, batch_size : int = 256, reward_scale : int = 2, tau = 0.005):
        self.tau = tau
        self.gamma = gamma
        
        self.memory = ReplayMemory(mem_size = mem_size, state_size = state_size, action_size = action_size)
        self.batch_size = batch_size
        self.action_size = action_size

        self.actor = ActorNetwork(lr = alpha, state_size = state_size, action_size = action_size, action_range = action_range)
        self.critic_1 = CriticNetwork(lr = beta, fc1_dims = fc1_dims, fc2_dims = fc2_dims, state_size = state_size, action_size = action_size)
        self.critic_2 = CriticNetwork(lr = beta, fc1_dims = fc1_dims, fc2_dims = fc2_dims, state_size = state_size, action_size = action_size)
        self.value = ValueNetwork(lr = beta, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)
        self.target_value = ValueNetwork(lr = beta, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)

        self.scale = reward_scale
        self.update_network_parameters(tau = 1) # Sets the values of the target value network equal to the ones of the value

    def choose_action(self, state):
        state = torch.Tensor(np.array([state])).to(self.actor.device)
        with torch.no_grad():
            actions, _ = self.actor.sample_normal(state, reparameterize = False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done, bc, imagined):
        self.memory.store_transition(state = state, action = action, reward = reward, new_state = new_state, done = done, bc = bc, imagined = imagined)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def get_batch(self):
        if self.memory.mem_counter < self.batch_size:
            return None, None, None, None, None, None, None

        state, action, reward, new_state, done, bc, imagined = self.memory.sample_buffer(batch_size = self.batch_size)
        state = torch.tensor(state, dtype = torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype = torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype = torch.float).to(self.actor.device)
        new_state = torch.tensor(new_state, dtype = torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        bc = torch.tensor(bc).to(self.actor.device)
        imagined = torch.tensor(imagined).to(self.actor.device)

        return state, action, reward, new_state, done, bc, imagined

    def learn_from_batch(self, state : torch.Tensor, action : torch.Tensor, reward : torch.Tensor, new_state : torch.Tensor, done : torch.Tensor, bc : torch.Tensor):
        # Calculate the value of states and new states according to the value and target value networks
        value = self.value(state).view(-1)
        target_value = self.target_value(new_state).view(-1)
        target_value[done] = 0.0 # Where new states are terminal, set value to 0, that is the definition of the value function

        # Obtain critic value under new policy, notice we use the action sampled from the new policy and not the ones from the batch
        actions, log_probs = self.actor.sample_normal(state, reparameterize = False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Train Value
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph = True)
        self.value.optimizer.step()

        # Train actor, resample actions with exploration, obtain the critic values and calculate the actor loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize = True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        sac_actor_loss = log_probs - critic_value
        sac_actor_loss = torch.mean(sac_actor_loss)
        bc_loss = torch.mean(bc * F.mse_loss(action, actions))
        actor_loss = bc_loss + sac_actor_loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()

        # Train critics, obtain critic values for old policy
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * target_value
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.value.eval()
        self.target_value.eval()

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.value.train()
        self.target_value.train()