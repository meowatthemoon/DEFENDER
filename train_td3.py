import argparse
import os

import gym

from models.td3_agent import TD3Agent
from hyper_params import HyperParams
from util.util import set_seed_everywhere
from util.episode import EpisodeHistory
from util.environments import valid_env_names

ALGORITHM_NAME = 'TD3'

def train_td3(env_name : str, n_episodes : int, seed : int, batch_size : int):
    assert env_name in valid_env_names, f"{env_name} is not valid environment name yet!"

    # Setup Directories
    result_dir = os.path.join(HyperParams.results_path, ALGORITHM_NAME, env_name)
    result_path = os.path.join(result_dir, f"seed_{seed}.json")
    os.makedirs(result_dir, exist_ok = True)

    demo_dir = os.path.join(HyperParams.demonstration_path, env_name)
    os.makedirs(demo_dir, exist_ok = True)

    # Setup Configs
    set_seed_everywhere(seed = seed)

    # Initialize
    env = gym.make(env_name)
    td3_agent = TD3Agent(
        state_size = env.observation_space.shape[0], 
        action_size = env.action_space.shape[0], 
        max_action = env.action_space.high,
        min_action = env.action_space.low,
        batch_size = batch_size
    )
    
    episode_history = EpisodeHistory()

    # Train Loop
    for episode in range(n_episodes):
        observation, info = env.reset()
        truncated = False
        terminated = False
        done = False
        acc_reward = 0
        n_steps = 0

        # Episode loop
        while not done:
            action = td3_agent.choose_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            td3_agent.remember(observation, action, reward, new_observation, done, bc = 0, imagined = 0)
            state_batch, action_batch, reward_batch, new_state_batch, done_batch, bc_batch, imagined = td3_agent.get_batch()
            if state_batch is not None:
                td3_agent.learn_from_batch(state = state_batch, action = action_batch, new_state = new_state_batch, reward = reward_batch, done = done_batch, bc = bc_batch)
            
            observation = new_observation
            acc_reward += reward
            n_steps += 1

        print(f"{ALGORITHM_NAME} #{seed}: {env_name} {episode + 1}/{n_episodes} | R = {acc_reward} | L = {n_steps}")
        crash = 1 if terminated and n_steps < env.spec.max_episode_steps else 0

        # Save episode results
        episode_history.store_episode(accumulate_reward = acc_reward, episode_length = n_steps, terminated = terminated, truncated = truncated, last_info = info, crash = crash)


    # Save run results
    episode_history.save_result(result_path = result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Experiment arguments')
    parser.add_argument('--env_name', type = str, help = 'Name of the task')
    parser.add_argument('--seed', type = int, help = 'Seed number.')
    parser.add_argument('--n_episodes', type = int, help = 'Number of training episodes.')
    parser.add_argument('--batch_size', type = int, help = 'Batch size of learning agent.')
    args = parser.parse_args()

    train_td3(
        env_name = args.env_name, 
        seed = args.seed, 
        n_episodes = args.n_episodes, 
        batch_size = args.batch_size, 
    )