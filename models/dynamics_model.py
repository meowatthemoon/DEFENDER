import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class DynamicsModel(nn.Module):
    def __init__(self, state_size : int, action_size : int, lr: float = 0.0003,  fc1_dims :int = 256, fc2_dims : int = 256):
        super(DynamicsModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.state_size + self.action_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, self.state_size)

        self.optimizer = Adam(self.parameters(), lr = lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_action_embed_1 = self.fc1(torch.cat([state, action], dim = 1))
        state_action_embed_1_act = F.relu(state_action_embed_1)

        state_action_embed_2 = self.fc2(state_action_embed_1_act)
        state_action_embed_2_act = F.relu(state_action_embed_2)

        new_state = self.q(state_action_embed_2_act)
        return new_state
    
    def learn_from_batch(self, states, actions, new_states, imagined):
        states = states[imagined == 0]
        actions = actions[imagined == 0]
        new_states = new_states[imagined == 0]
        self.optimizer.zero_grad()
        new_states_pred = self.forward(state = states, action = actions)
        loss = F.mse_loss(new_states_pred, new_states)
        #loss = loss * (1 - imagined)
        loss.backward()
        self.optimizer.step()
