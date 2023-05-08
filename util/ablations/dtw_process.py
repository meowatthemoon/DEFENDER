import os
import json

import numpy as np

def get_score(experiment_file : str, safe_key : str, unsafe_key : str):
    horizon = 1000
    with open(experiment_file, 'r') as f:
        dtw_json = json.load(f)

    episode_percs = []
    crash_avoided = []
    for episode in dtw_json:
        filter_idx = None

        safe_transitions = episode[safe_key]
        unsafe_transitions = episode[unsafe_key]
        episode_length = len(safe_transitions)

        for t_idx  in range(episode_length):
            difference = safe_transitions[t_idx] - unsafe_transitions[t_idx]
            if filter_idx is None and  difference > 0:
                filter_idx = t_idx
        episode_perc = 1 if filter_idx is None else filter_idx / episode_length
        episode_percs.append(episode_perc)
        if episode_length < horizon:
            crash_avoided.append(0 if episode_perc == 1 else 1)

    score = np.mean(episode_percs) 
    if len(crash_avoided) > 0:
        score = score * np.mean(crash_avoided)
    return score

def main(safe_keys : list, experiment_files : list):
    unsafe_keys = [safe_key.replace("safe", "unsafe") for safe_key in safe_keys]
    base_dir = "dtws"
    results = []
    # Loop through each key, will be the same for safe and unsafe
    for safe_key in safe_keys:
        for unsafe_key in unsafe_keys:
            # Loop through each experiment and get its results
            mean_score = np.mean([get_score(experiment_file = os.path.join(base_dir, experiment_file), safe_key = safe_key, unsafe_key = unsafe_key) for experiment_file in experiment_files])
            results.append([safe_key, unsafe_key, mean_score])
    results.sort(key=lambda x: x[2], reverse=True) 
    for r in results:
        if r[2] > 0.1:
            print(f"{'' if r[0] != r[1].replace('unsafe', 'safe') else '(!)'}{r[0]} vs {r[1]} | {r[2]}")
if __name__ == '__main__':
    # Get keys
    safe_keys = [
        "min_safe_full_dtws",
        "max_safe_full_dtws",
        "mean_safe_full_dtws",
        "min_safe_same_dtws", 
        "max_safe_same_dtws", 
        "mean_safe_same_dtws",
        "min_safe_bothwindow5_dtws", 
        "max_safe_bothwindow5_dtws", 
        "mean_safe_bothwindow5_dtws",
        "min_safe_bothwindow10_dtws", 
        "max_safe_bothwindow10_dtws", 
        "mean_safe_bothwindow10_dtws",
        "min_safe_demowindow5_dtws", 
        "max_safe_demowindow5_dtws", 
        "mean_safe_demowindow5_dtws",
        "min_safe_demowindow10_dtws", 
        "max_safe_demowindow10_dtws", 
        "mean_safe_demowindow10_dtws",
        "min_safe_trajwindow5_dtws", 
        "max_safe_trajwindow5_dtws", 
        "mean_safe_trajwindow5_dtws",
        "min_safe_trajwindow10_dtws", 
        "max_safe_trajwindow10_dtws", 
        "mean_safe_trajwindow10_dtws"
    ]

    state_experiment_files = [
        #"dtw_Ant-v4_state_50.json", # TODO
        "dtw_Hopper-v4_state_50.json",
        "dtw_InvertedDoublePendulum-v4_state_42.json",
        "dtw_Walker2d-v4_state_50.json"
    ]

    state_action_experiment_files = [
        #"dtw_Ant-v4_state_action_50.json", # TODO
        "dtw_Hopper-v4_state_action_50.json",
        "dtw_InvertedDoublePendulum-v4_state_action_42.json",
        #"dtw_Walker2d-v4_state_action_50.json" # TODO
    ]

    """
    main(safe_keys = safe_keys, experiment_files = state_experiment_files)
    print("\n\n\n\n\n#################### \n\n\n\n\n\n\n")
    main(safe_keys = safe_keys, experiment_files = state_action_experiment_files)
    """
    main(safe_keys = safe_keys, experiment_files = [state_experiment_files[0]])
    print("\n\n\n\n\n#################### \n\n\n\n\n\n\n")
    main(safe_keys = safe_keys, experiment_files = [state_experiment_files[1]])
    print("\n\n\n\n\n#################### \n\n\n\n\n\n\n")
    main(safe_keys = safe_keys, experiment_files = [state_experiment_files[2]])
