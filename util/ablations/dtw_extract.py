import argparse
import json
from threading import Thread
import os

from fastdtw import fastdtw
import gym
import numpy as np

from util.demonstration import sample_demonstrations

np.random.seed(0)

N_THREADS = 10
MAX_EPISODES = 500
MAX_LENGTH = 20


def calculate_dtws(trajectories : np.array, safe_demonstrations : np.array, unsafe_demonstrations : np.array, results_list : list, start_idx : int):
    n_trajectories = trajectories.shape[0]
    ep_idx = start_idx

    while ep_idx < n_trajectories:
        episode = trajectories[ep_idx]
        episode_length = episode.shape[0]

        # ------------------------------------------------------------------------------------------------------ #
        min_safe_full_dtws, max_safe_full_dtws, mean_safe_full_dtws = [], [], []
        min_unsafe_full_dtws, max_unsafe_full_dtws, mean_unsafe_full_dtws = [], [], []

        min_safe_same_dtws, max_safe_same_dtws, mean_safe_same_dtws = [], [], []
        min_unsafe_same_dtws, max_unsafe_same_dtws, mean_unsafe_same_dtws = [], [], []

        min_safe_bothwindow5_dtws, max_safe_bothwindow5_dtws, mean_safe_bothwindow5_dtws = [], [], []
        min_unsafe_bothwindow5_dtws, max_unsafe_bothwindow5_dtws, mean_unsafe_bothwindow5_dtws = [], [], []

        min_safe_bothwindow10_dtws, max_safe_bothwindow10_dtws, mean_safe_bothwindow10_dtws = [], [], []
        min_unsafe_bothwindow10_dtws, max_unsafe_bothwindow10_dtws, mean_unsafe_bothwindow10_dtws = [], [], []

        min_safe_demowindow5_dtws, max_safe_demowindow5_dtws, mean_safe_demowindow5_dtws = [], [], []
        min_unsafe_demowindow5_dtws, max_unsafe_demowindow5_dtws, mean_unsafe_demowindow5_dtws = [], [], []

        min_safe_demowindow10_dtws, max_safe_demowindow10_dtws, mean_safe_demowindow10_dtws = [], [], []
        min_unsafe_demowindow10_dtws, max_unsafe_demowindow10_dtws, mean_unsafe_demowindow10_dtws = [], [], []

        min_safe_trajwindow5_dtws, max_safe_trajwindow5_dtws, mean_safe_trajwindow5_dtws = [], [], []
        min_unsafe_trajwindow5_dtws, max_unsafe_trajwindow5_dtws, mean_unsafe_trajwindow5_dtws = [], [], []

        min_safe_trajwindow10_dtws, max_safe_trajwindow10_dtws, mean_safe_trajwindow10_dtws = [], [], []
        min_unsafe_trajwindow10_dtws, max_unsafe_trajwindow10_dtws, mean_unsafe_trajwindow10_dtws = [], [], []

        # ------------------------------------------------------------------------------------------------------ #

        for transition_idx in range(episode_length):
            trajectory = episode[:transition_idx + 1]
            trajectory = trajectory[-MAX_LENGTH:]
            length = trajectory.shape[0]

            safe_full_dtws = [fastdtw(demonstration, trajectory)[0] for demonstration in safe_demonstrations]
            min_safe_full_dtws.append(np.min(safe_full_dtws)), max_safe_full_dtws.append(np.max(safe_full_dtws)), mean_safe_full_dtws.append(np.mean(safe_full_dtws))
            unsafe_full_dtws = [fastdtw(demonstration, trajectory)[0] for demonstration in unsafe_demonstrations]
            min_unsafe_full_dtws.append(np.min(unsafe_full_dtws)), max_unsafe_full_dtws.append(np.max(unsafe_full_dtws)), mean_unsafe_full_dtws.append(np.mean(unsafe_full_dtws))

            safe_same_dtws = [fastdtw(demonstration[-length:], trajectory)[0] for demonstration in safe_demonstrations]
            min_safe_same_dtws.append(np.min(safe_same_dtws)), max_safe_same_dtws.append(np.max(safe_same_dtws)), mean_safe_same_dtws.append(np.mean(safe_same_dtws))
            unsafe_same_dtws = [fastdtw(demonstration[-length:], trajectory)[0] for demonstration in unsafe_demonstrations]
            min_unsafe_same_dtws.append(np.min(unsafe_same_dtws)), max_unsafe_same_dtws.append(np.max(unsafe_same_dtws)), mean_unsafe_same_dtws.append(np.mean(unsafe_same_dtws))

            safe_bothwindow5_dtws = [fastdtw(demonstration[-5:], trajectory[-5:])[0] for demonstration in safe_demonstrations]
            min_safe_bothwindow5_dtws.append(np.min(safe_bothwindow5_dtws)), max_safe_bothwindow5_dtws.append(np.max(safe_bothwindow5_dtws)), mean_safe_bothwindow5_dtws.append(np.mean(safe_bothwindow5_dtws))
            unsafe_bothwindow5_dtws = [fastdtw(demonstration[-5:], trajectory[-5:])[0] for demonstration in unsafe_demonstrations]
            min_unsafe_bothwindow5_dtws.append(np.min(unsafe_bothwindow5_dtws)), max_unsafe_bothwindow5_dtws.append(np.max(unsafe_bothwindow5_dtws)), mean_unsafe_bothwindow5_dtws.append(np.mean(unsafe_bothwindow5_dtws))

            safe_bothwindow10_dtws = [fastdtw(demonstration[-10:], trajectory[-10:])[0] for demonstration in safe_demonstrations]
            min_safe_bothwindow10_dtws.append(np.min(safe_bothwindow10_dtws)), max_safe_bothwindow10_dtws.append(np.max(safe_bothwindow10_dtws)), mean_safe_bothwindow10_dtws.append(np.mean(safe_bothwindow10_dtws))
            unsafe_bothwindow10_dtws = [fastdtw(demonstration[-10:], trajectory[-10:])[0] for demonstration in unsafe_demonstrations]
            min_unsafe_bothwindow10_dtws.append(np.min(unsafe_bothwindow10_dtws)), max_unsafe_bothwindow10_dtws.append(np.max(unsafe_bothwindow10_dtws)), mean_unsafe_bothwindow10_dtws.append(np.mean(unsafe_bothwindow10_dtws))

            ####
            safe_demowindow5_dtws = [fastdtw(demonstration[-5:], trajectory)[0] for demonstration in safe_demonstrations]
            min_safe_demowindow5_dtws.append(np.min(safe_demowindow5_dtws)), max_safe_demowindow5_dtws.append(np.max(safe_demowindow5_dtws)), mean_safe_demowindow5_dtws.append(np.mean(safe_demowindow5_dtws))
            unsafe_demowindow5_dtws = [fastdtw(demonstration[-5:], trajectory)[0] for demonstration in unsafe_demonstrations]
            min_unsafe_demowindow5_dtws.append(np.min(unsafe_demowindow5_dtws)), max_unsafe_demowindow5_dtws.append(np.max(unsafe_demowindow5_dtws)), mean_unsafe_demowindow5_dtws.append(np.mean(unsafe_demowindow5_dtws))

            safe_demowindow10_dtws = [fastdtw(demonstration[-10:], trajectory)[0] for demonstration in safe_demonstrations]
            min_safe_demowindow10_dtws.append(np.min(safe_demowindow10_dtws)), max_safe_demowindow10_dtws.append(np.max(safe_demowindow10_dtws)), mean_safe_demowindow10_dtws.append(np.mean(safe_demowindow10_dtws))
            unsafe_demowindow10_dtws = [fastdtw(demonstration[-10:], trajectory)[0] for demonstration in unsafe_demonstrations]
            min_unsafe_demowindow10_dtws.append(np.min(unsafe_demowindow10_dtws)), max_unsafe_demowindow10_dtws.append(np.max(unsafe_demowindow10_dtws)), mean_unsafe_demowindow10_dtws.append(np.mean(unsafe_demowindow10_dtws))

            ###
            safe_trajwindow5_dtws = [fastdtw(demonstration, trajectory[-5:])[0] for demonstration in safe_demonstrations]
            min_safe_trajwindow5_dtws.append(np.min(safe_trajwindow5_dtws)), max_safe_trajwindow5_dtws.append(np.max(safe_trajwindow5_dtws)), mean_safe_trajwindow5_dtws.append(np.mean(safe_trajwindow5_dtws))
            unsafe_trajwindow5_dtws = [fastdtw(demonstration, trajectory[-5:])[0] for demonstration in unsafe_demonstrations]
            min_unsafe_trajwindow5_dtws.append(np.min(unsafe_trajwindow5_dtws)), max_unsafe_trajwindow5_dtws.append(np.max(unsafe_trajwindow5_dtws)), mean_unsafe_trajwindow5_dtws.append(np.mean(unsafe_trajwindow5_dtws))

            safe_trajwindow10_dtws = [fastdtw(demonstration, trajectory[-10:])[0] for demonstration in safe_demonstrations]
            min_safe_trajwindow10_dtws.append(np.min(safe_trajwindow10_dtws)), max_safe_trajwindow10_dtws.append(np.max(safe_trajwindow10_dtws)), mean_safe_trajwindow10_dtws.append(np.mean(safe_trajwindow10_dtws))
            unsafe_trajwindow10_dtws = [fastdtw(demonstration, trajectory[-10:])[0] for demonstration in unsafe_demonstrations]
            min_unsafe_trajwindow10_dtws.append(np.min(unsafe_trajwindow10_dtws)), max_unsafe_trajwindow10_dtws.append(np.max(unsafe_trajwindow10_dtws)), mean_unsafe_trajwindow10_dtws.append(np.mean(unsafe_trajwindow10_dtws))
            print(f"Episode {ep_idx + 1}/{n_trajectories} | Transition {transition_idx + 1}/ {episode_length}")

        episode_results = {
            "ep_idx" : ep_idx,
            "length" : length,
            "min_safe_full_dtws" : min_safe_full_dtws,
            "max_safe_full_dtws" : max_safe_full_dtws,
            "mean_safe_full_dtws" : mean_safe_full_dtws,
            "min_unsafe_full_dtws" : min_unsafe_full_dtws,
            "max_unsafe_full_dtws" : max_unsafe_full_dtws,
            "mean_unsafe_full_dtws" : mean_unsafe_full_dtws,

            "min_safe_same_dtws" : min_safe_same_dtws, 
            "max_safe_same_dtws" : max_safe_same_dtws, 
            "mean_safe_same_dtws" : mean_safe_same_dtws,
            "min_unsafe_same_dtws" : min_unsafe_same_dtws, 
            "max_unsafe_same_dtws" : max_unsafe_same_dtws, 
            "mean_unsafe_same_dtws" : mean_unsafe_same_dtws,

            "min_safe_bothwindow5_dtws" : min_safe_bothwindow5_dtws, 
            "max_safe_bothwindow5_dtws" : max_safe_bothwindow5_dtws, 
            "mean_safe_bothwindow5_dtws" : mean_safe_bothwindow5_dtws,
            "min_unsafe_bothwindow5_dtws" : min_unsafe_bothwindow5_dtws, 
            "max_unsafe_bothwindow5_dtws" : max_unsafe_bothwindow5_dtws, 
            "mean_unsafe_bothwindow5_dtws" : mean_unsafe_bothwindow5_dtws,

            "min_safe_bothwindow10_dtws" : min_safe_bothwindow10_dtws, 
            "max_safe_bothwindow10_dtws" : max_safe_bothwindow10_dtws, 
            "mean_safe_bothwindow10_dtws" : mean_safe_bothwindow10_dtws,
            "min_unsafe_bothwindow10_dtws" : min_unsafe_bothwindow10_dtws, 
            "max_unsafe_bothwindow10_dtws" : max_unsafe_bothwindow10_dtws, 
            "mean_unsafe_bothwindow10_dtws" : mean_unsafe_bothwindow10_dtws,


            "min_safe_demowindow5_dtws" : min_safe_demowindow5_dtws, 
            "max_safe_demowindow5_dtws" : max_safe_demowindow5_dtws, 
            "mean_safe_demowindow5_dtws" : mean_safe_demowindow5_dtws,
            "min_unsafe_demowindow5_dtws" : min_unsafe_demowindow5_dtws, 
            "max_unsafe_demowindow5_dtws" : max_unsafe_demowindow5_dtws, 
            "mean_unsafe_demowindow5_dtws" : mean_unsafe_demowindow5_dtws,

            "min_safe_demowindow10_dtws" : min_safe_demowindow10_dtws, 
            "max_safe_demowindow10_dtws" : max_safe_demowindow10_dtws, 
            "mean_safe_demowindow10_dtws" : mean_safe_demowindow10_dtws,
            "min_unsafe_demowindow10_dtws" : min_unsafe_demowindow10_dtws, 
            "max_unsafe_demowindow10_dtws" : max_unsafe_demowindow10_dtws, 
            "mean_unsafe_demowindow10_dtws" : mean_unsafe_demowindow10_dtws,


            "min_safe_trajwindow5_dtws" : min_safe_trajwindow5_dtws, 
            "max_safe_trajwindow5_dtws" : max_safe_trajwindow5_dtws, 
            "mean_safe_trajwindow5_dtws" : mean_safe_trajwindow5_dtws,
            "min_unsafe_trajwindow5_dtws" : min_unsafe_trajwindow5_dtws, 
            "max_unsafe_trajwindow5_dtws" : max_unsafe_trajwindow5_dtws, 
            "mean_unsafe_trajwindow5_dtws" : mean_unsafe_trajwindow5_dtws,

            "min_safe_trajwindow10_dtws" : min_safe_trajwindow10_dtws, 
            "max_safe_trajwindow10_dtws" : max_safe_trajwindow10_dtws, 
            "mean_safe_trajwindow10_dtws" : mean_safe_trajwindow10_dtws,
            "min_unsafe_trajwindow10_dtws" : min_unsafe_trajwindow10_dtws, 
            "max_unsafe_trajwindow10_dtws" : max_unsafe_trajwindow10_dtws, 
            "mean_unsafe_trajwindow10_dtws" : mean_unsafe_trajwindow10_dtws,
        }

        results_list[ep_idx] = episode_results
        ep_idx += N_THREADS

def test(safe_demonstrations : np.array, unsafe_demonstrations : np.array, trajectories : np.array, env_name : str, transition_type : str, n_demos : int):
    n_trajectories = trajectories.shape[0]
    results = [{} for i in range(n_trajectories)]

    threads = [Thread(target=calculate_dtws, args=(trajectories, safe_demonstrations, unsafe_demonstrations, results, start_idx)) for start_idx in range(N_THREADS)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Save results 
    base_dir = './dtws'
    os.makedirs(base_dir, exist_ok = True)
    with open(f"{base_dir}/dtw_{env_name}_{transition_type}_{n_demos}.json", 'w') as outfile:
        json.dump(results, outfile, indent = 4)

def main(env_name : str, transition_type : str, n_demos : int):
    env = gym.make(env_name)
    horizon = env.spec.max_episode_steps

    # Load demonstrations
    demo_dir = f"demonstrations/{env_name}"
    demo_files = sorted(os.listdir(demo_dir))
    safe_demonstrations = []
    unsafe_demonstrations = []
    for demo_file in demo_files:
        demonstration_path = os.path.join(demo_dir, demo_file)
        with open(demonstration_path, "r") as f:
            demonstration_json = json.load(f)

            length = len(demonstration_json)
            states = [transition["state"] for transition in demonstration_json]
            actions = [transition["action"] for transition in demonstration_json]

            if transition_type == 'state':
                demonstration = np.array([state for state in states])
            elif transition_type == 'state_action':
                demonstration = np.array([state+action for state, action in zip(states, actions)])
            else:
                raise ValueError
            
            if length == horizon:
                safe_demonstrations.append(demonstration)
            elif length < horizon:
                unsafe_demonstrations.append(demonstration)
            else:
                raise ValueError
    safe_demonstrations = np.array(safe_demonstrations)
    #safe_demonstrations = safe_demonstrations[np.random.choice(safe_demonstrations.shape[0], N_DEMOS, replace = False)]
    safe_demonstrations = sample_demonstrations(demonstrations = safe_demonstrations, n_samples = n_demos)
    unsafe_demonstrations = np.array(unsafe_demonstrations, dtype = object)
    #unsafe_demonstrations = unsafe_demonstrations[np.random.choice(unsafe_demonstrations.shape[0], N_DEMOS, replace = False)]]
    unsafe_demonstrations = sample_demonstrations(demonstrations = unsafe_demonstrations, n_samples = n_demos)

    # Load episode memory
    trajectories = []
    with open(f"results/SAC/{env_name}/memory_0.json", "r") as f:
        episodes = json.load(f)
        for episode in episodes[:MAX_EPISODES]:
            states = episode["states"]
            actions = episode["actions"]

            if transition_type == 'state':
                trajectory = np.array([state for state in states])
            elif transition_type == 'state_action':
                trajectory = np.array([state+action for state, action in zip(states, actions)])
            else:
                raise ValueError
            trajectories.append(trajectory)
    trajectories = np.array(trajectories, dtype = object)

    test(safe_demonstrations = safe_demonstrations, unsafe_demonstrations = unsafe_demonstrations, trajectories = trajectories, env_name = env_name, transition_type = transition_type, n_demos = n_demos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Experiment arguments')
    parser.add_argument('--env_name', type = str, help = 'Name of the task')
    args = parser.parse_args()
    for transition_type in ["state", "state_action"]:
        for n_demos in [10, 20, 50]:
            # TODO remove this block
            if args.env_name == 'InvertedDoublePendulum-v4':
                if n_demos <= 20 and transition_type == "state":
                    continue
                if n_demos == 50:
                    n_demos = 42
            # TODO until here
            main(env_name = args.env_name, transition_type = transition_type, n_demos = n_demos)