"""
Adapted from Phil Tabor's (Machine Learning with Phil) TD3 implementation.
https://github.com/philtabor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from models.memory import ReplayMemory

class CriticNetwork(nn.Module):
    def __init__(self, beta : float, state_size : int, fc1_dims : int, fc2_dims : int, action_size : int):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)

        self.optimizer = Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(torch.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1
    
class ActorNetwork(nn.Module):
    def __init__(self, alpha : float, state_size : int, fc1_dims : int, fc2_dims : int, action_size : int):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, action_size)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = torch.tanh(self.mu(prob))

        return mu
    
class TD3Agent():
    def __init__(self, state_size : int, action_size : int, max_action, min_action, alpha : float = 0.0003, beta : float = 0.0003, gamma : float = 0.99, mem_size : int = 1000000, fc1_dims : int = 256, fc2_dims : int = 256, batch_size : int = 256, reward_scale : int = 2, tau = 0.005, update_actor_interval=2, warmup=1000, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.min_action = min_action
        self.memory = ReplayMemory(mem_size = mem_size, state_size = state_size, action_size = action_size)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.action_size = action_size
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha = alpha, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)

        self.critic_1 = CriticNetwork(beta = beta, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)
        self.critic_2 = CriticNetwork(beta, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)

        self.target_actor = ActorNetwork(alpha = alpha, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)
        self.target_critic_1 = CriticNetwork(beta = beta, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)
        self.target_critic_2 = CriticNetwork(beta = beta, action_size = action_size, state_size = state_size, fc1_dims = fc1_dims, fc2_dims = fc2_dims)

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size = (self.action_size,))).to(self.actor.device)
        else:
            state = torch.tensor(observation, dtype = torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + torch.tensor(np.random.normal(scale = self.noise), dtype = torch.float).to(self.actor.device)

        mu_prime = torch.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done, bc, imagined):
        self.memory.store_transition(state = state, action = action, reward = reward, new_state = new_state, done = done, bc = bc, imagined = imagined)

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
        target_actions = self.target_actor.forward(new_state)
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])
        
        q1_ = self.target_critic_1.forward(new_state, target_actions)
        q2_ = self.target_critic_2.forward(new_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = torch.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)