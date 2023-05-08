import argparse
import json
import os

import gym
import numpy as np
import torch

from hyper_params import HyperParams
from models.dynamics_model import DynamicsModel
from models.sac_agent import SACAgent
from util.environments import valid_env_names
from util.episode import EpisodeHistory
from util.util import set_seed_everywhere
from util.demonstration import load_demonstrations
from util.dtw_functions import get_function_by_key

ALGORITHM_NAME = 'SAC_DEFENDER'

def train_sac(env_name : str, n_episodes : int, seed : int, batch_size : int, n_demos : int, dtw_safe_key : str, dtw_unsafe_key : str, save_every : int, traj_type : str):
    assert env_name in valid_env_names, f"{env_name} is not valid environment name yet!"

    dtw_safe_function = get_function_by_key(key = dtw_safe_key)
    dtw_unsafe_function = get_function_by_key(key = dtw_unsafe_key)

    # Setup Directories
    result_dir = os.path.join(HyperParams.results_path, ALGORITHM_NAME, env_name)
    result_path = os.path.join(result_dir, f"seed_{seed}.json")
    os.makedirs(result_dir, exist_ok = True)

    # Setup Configs
    set_seed_everywhere(seed = seed)

    # Initialize
    env = gym.make(env_name)
    horizon = env.spec.max_episode_steps
    sac_agent = SACAgent(
        state_size = env.observation_space.shape[0], 
        action_size = env.action_space.shape[0], 
        action_range = env.action_space.high,
        batch_size = batch_size
    )
    dynamics_model = DynamicsModel(state_size = env.observation_space.shape[0], action_size = env.action_space.shape[0])

    episode_history = EpisodeHistory()

    # Load demonstrations into memory
    demo_dir = os.path.join(HyperParams.demonstration_path, env_name)
    demo_files = sorted(os.listdir(demo_dir))
    for demo_file in demo_files:
        demonstration_path = os.path.join(demo_dir, demo_file)
        with open(demonstration_path, "r") as f:
            demonstration_json = json.load(f)
            episode_length = len(demonstration_json)
            for transition in demonstration_json:
                sac_agent.remember(
                    state = transition["state"], 
                    action = transition["action"], 
                    reward = transition["reward"], 
                    new_state = transition["new_state"], 
                    done = transition["done"], 
                    bc = int(episode_length == horizon), 
                    imagined = 0
                )

    # Load demonstrations for DTW
    min_reward, safe_demonstrations, unsafe_demonstrations = load_demonstrations(env_name = env_name, transition_type = traj_type, horizon = horizon, n_demos = n_demos)

    # Train Loop
    for episode in range(n_episodes):
        observation, info = env.reset()
        truncated = False
        terminated = False
        done = False
        acc_reward = 0
        n_steps = 0
        filtered = False

        trajectory = []

        # Episode loop
        while not done:
            action = sac_agent.choose_action(observation)
            if traj_type == 'state':
                trajectory.append(observation)
            else:
                trajectory.append(list(observation) + list(action))

            safe_dtw = dtw_safe_function(trajectory = np.array(trajectory), demonstrations = safe_demonstrations)
            unsafe_dtw = dtw_unsafe_function(trajectory = np.array(trajectory), demonstrations = unsafe_demonstrations)
            filtered = safe_dtw > unsafe_dtw
            if filtered:
                with torch.no_grad():
                    new_observation = dynamics_model.forward(state = torch.Tensor([observation]).to(dynamics_model.device), action = torch.Tensor([action]).to(dynamics_model.device))
                    new_observation = new_observation.cpu().detach().numpy()[0]
                    reward = min_reward
                    terminated = False
                    truncated = False
                    done = True
                    info = {"safe_dtw" : safe_dtw, "unsafe_dtw" : unsafe_dtw}
            else:
                new_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            sac_agent.remember(observation, action, reward, new_observation, done, bc = 0, imagined = int(filtered))
            state_batch, action_batch, reward_batch, new_state_batch, done_batch, bc_batch, imagined_batch = sac_agent.get_batch()
            if state_batch is not None:
                sac_agent.learn_from_batch(state = state_batch, action = action_batch, new_state = new_state_batch, reward = reward_batch, done = done_batch, bc = bc_batch)
                dynamics_model.learn_from_batch(states = state_batch, actions = action_batch, new_states = new_state_batch, imagined = imagined_batch)

            observation = new_observation
            acc_reward += reward
            n_steps += 1

        print(f"{ALGORITHM_NAME} #{seed} {dtw_safe_key} vs {dtw_unsafe_key}: {env_name} {episode + 1}/{n_episodes} | R = {acc_reward} | L = {n_steps} | Filtered? {filtered}")
        crash = 1 if terminated and n_steps < env.spec.max_episode_steps else 0

        # Save episode results
        episode_history.store_episode(accumulate_reward = acc_reward, episode_length = n_steps, terminated = terminated, truncated = truncated, last_info = info, crash = crash, filtered = filtered)

    # Save run results
    episode_history.save_result(result_path = result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Experiment arguments')
    parser.add_argument('--env_name', type = str, help = 'Name of the task')
    parser.add_argument('--seed', type = int, help = 'Seed number.')
    parser.add_argument('--n_episodes', type = int, help = 'Number of training episodes.')
    parser.add_argument('--batch_size', type = int, help = 'Batch size of learning agent.')
    parser.add_argument('--n_demos', type = int, help = 'Max number of demonstrations for each type (safe/unsafe).')
    parser.add_argument('--dtw_safe', type = str, help = 'DTW Type for safe demos')
    parser.add_argument('--dtw_unsafe', type = str, help = 'DTW Type for unsafe demos')
    parser.add_argument('--traj_type', type = str, default = 'state', help = 'state or state_action')
    args = parser.parse_args()

    
    train_sac(
        env_name = args.env_name, 
        seed = args.seed, 
        n_episodes = args.n_episodes, 
        batch_size = args.batch_size, 
        n_demos = args.n_demos, 
        dtw_safe_key = args.dtw_safe, 
        dtw_unsafe_key = args.dtw_unsafe,
        save_every = HyperParams.save_every,
        traj_type = args.traj_type
    )