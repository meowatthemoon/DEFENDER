import numpy as np


class  ReplayMemory:
    def __init__(self, mem_size : int, state_size : int, action_size : int):
        self.mem_size = mem_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, state_size))
        self.new_state_memory = np.zeros((self.mem_size, state_size))
        self.action_memory = np.zeros((self.mem_size, action_size))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        self.bc_memory = np.zeros(self.mem_size)
        self.imagined_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, new_state, done, bc, imagined):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.bc_memory[index] = bc
        self.imagined_memory[index] = imagined

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        bc = self.bc_memory[batch]
        imagined = self.imagined_memory[batch]

        return states, actions, rewards, new_states, dones, bc, imagined