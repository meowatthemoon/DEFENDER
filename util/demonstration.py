import json
import os

from fastdtw import fastdtw
import numpy as np

from models.sac_agent import SACAgent
from hyper_params import HyperParams

def demonstrate(agent : SACAgent, env):
    demonstration = []
    observation, info = env.reset()
    done = False
    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transition = {
            "state" : [float(v) for v in observation],
            "action" : [float(v) for v in action],
            "new_state" : [float(v) for v in new_observation],
            "reward" : float(reward),
            "done" : int(done)
        }
        demonstration.append(transition)
        observation = new_observation

    return demonstration

def demonstration_to_trajectory(demonstration):
    trajectory = []
    for i in range(len(demonstration)):
        trajectory.append(demonstration[i]["state"] + demonstration[i]["action"])
    return trajectory

def load_safe_unsafe_trajectories(demo_path : str, number : int):
    demo_files = sorted(os.listdir(demo_path))
    N = len(demo_files)

    # Calculate the acc. rewards and sort demo index by acc. reward inc
    acc_rewards = []
    demonstrations = []
    for i in range(len(demo_files)):
        demonstration_path = os.path.join(demo_path, demo_files[i])
        with open(demonstration_path, "r") as f:
            demonstration = json.load(f)
            demonstrations.append(demonstration)
            acc_rewards.append(sum([demonstration[i]["reward"] for i in range(len(demonstration))]))
    demonstrations = np.array(demonstrations)
    
    acc_rewards = np.array(acc_rewards)
    indexes = np.argsort(acc_rewards)

    safe_indexes = indexes[-number:]
    unsafe_indexes = indexes[:number]
    return demonstrations[safe_indexes], demonstrations[unsafe_indexes]

def load_demonstrations(env_name : str, transition_type : str, horizon : int, n_demos : int):
    # Load demonstrations
    demo_dir = os.path.join(HyperParams.demonstration_path, env_name)
    demo_files = sorted(os.listdir(demo_dir))
    safe_demonstrations = []
    unsafe_demonstrations = []
    min_reward = float("inf")
    for demo_file in demo_files:
        demonstration_path = os.path.join(demo_dir, demo_file)
        with open(demonstration_path, "r") as f:
            demonstration_json = json.load(f)

            length = len(demonstration_json)
            states, actions = [], []
            for transition in demonstration_json:
                min_reward = min(min_reward, transition["reward"])
                states.append(transition["state"])
                actions.append(transition["action"])

            if transition_type == 'state':
                demonstration = np.array([state for state in states])
            elif transition_type == 'state_action':
                demonstration = np.array([list(state)+list(action) for state, action in zip(states, actions)])
            else:
                print(f"Length = {len(transition_type)}")
                print(f"Invalid transition type : [{transition_type}]_")
                print(f"Type : {type(transition_type)}")
                input(f"Equals : {transition_type == 'state_action'}")
                raise ValueError
            
            if length == horizon:
                safe_demonstrations.append(demonstration)
            elif length < horizon:
                unsafe_demonstrations.append(demonstration)
            else:
                raise ValueError
    safe_demonstrations = np.array(safe_demonstrations)
    safe_demonstrations = sample_demonstrations(demonstrations = safe_demonstrations, n_samples = n_demos)
    
    unsafe_demonstrations = np.array(unsafe_demonstrations, dtype = object)
    unsafe_demonstrations = sample_demonstrations(demonstrations = unsafe_demonstrations, n_samples = n_demos)

    return min_reward, safe_demonstrations, unsafe_demonstrations

def sample_demonstrations(demonstrations : np.array, n_samples : int):
    n_demos = demonstrations.shape[0]
    #assert n_demos >= n_samples, "N samples > N demos"
    if n_samples > n_demos:
        return demonstrations

    cost_table = [[None for i in range(n_demos)] for j in range(n_demos)]

    # Fill cost table
    worst, worst_r, worst_c = 0, -1, -1
    for row in range(n_demos):
        for col in range(row + 1, n_demos):
            cost = fastdtw(demonstrations[row], demonstrations[col])[0]
            cost_table[row][col] = cost
            cost_table[col][row] = cost
            if cost > worst:
                worst = cost
                worst_r = row
                worst_c = col
    samples = [worst_r, worst_c]

    # Find next worst aligned row
    while len(samples) < n_samples:
        worst, worst_r = 0, -1
        for row in range(n_demos):
            if row in samples:
                continue
            costs = [cost_table[row][col] for col in samples]
            cost = np.mean(costs)
            if cost > worst:
                worst = cost
                worst_r = row
        if worst_r == -1:
            raise Exception
        samples.append(worst_r)

    # Return
    samples = np.array(samples)
    return demonstrations[samples]