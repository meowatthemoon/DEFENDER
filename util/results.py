import pandas as pd
import numpy as np

import json
import os
from hyper_params import HyperParams

def results_standard(alg_name : str):
    results = []
    for task_name in os.listdir(os.path.join(HyperParams.results_path, alg_name)):
        task_crashes, task_acc_reward = [], []
        for experiment_file in os.listdir(os.path.join(HyperParams.results_path, alg_name, task_name)):
            with open(os.path.join(HyperParams.results_path, alg_name, task_name, experiment_file), 'r') as f:
                result_json = json.load(f)

                exp_rewards = result_json["rewards"]
                assert len(exp_rewards) == HyperParams.num_episode, f"{alg_name} {task_name} {experiment_file} : {len(exp_rewards)}"
                exp_crashes = result_json["crashes"]
                assert len(exp_crashes) == HyperParams.num_episode, f"{alg_name} {task_name} {experiment_file} : {len(exp_crashes)}"

                task_crashes.append(sum(exp_crashes) / len(exp_crashes) * 100)
                task_acc_reward.append(max(exp_rewards))

        task_crashes = np.array(task_crashes)
        task_acc_reward = np.array(task_acc_reward)

        mean_acc_reward = round(np.mean(task_acc_reward))
        std_acc_reward = round(np.std(task_acc_reward), 1)
        mean_crashes = round(np.mean(task_crashes))
        std_crashes = round(np.std(task_crashes), 1)

        row = [alg_name, task_name, mean_acc_reward, std_acc_reward, mean_crashes, std_crashes]
        results.append(row)

    df = pd.DataFrame(results, columns = ["Algorithm", "Task", "Acc. Reward", "Std. Reward", "Crashes", "Std. Crashes"])
    df.to_csv(f'{alg_name}.csv') 


def results_DEFENDER(alg_name : str):
    crash_results = {}
    reward_results = {}

    for task_name in os.listdir(os.path.join(HyperParams.results_path, alg_name)):
        for file in os.listdir(os.path.join(HyperParams.results_path, alg_name, task_name)):
            traj_type = "state_action" if "state_action" in file else "state"
            filters = file[:file.find("_state")].split("_vs_")
            safe_filter = filters[0]
            unsafe_filter = filters[1]

            with open(os.path.join(HyperParams.results_path, alg_name, task_name, file), 'r') as f:
                result_json = json.load(f)

            exp_rewards = result_json["rewards"]
            exp_crashes = result_json["crashes"]

            key = f"{task_name}|{traj_type}|{safe_filter}|{unsafe_filter}"
            if key in crash_results.keys():
                crash_results[key].append(sum(exp_crashes) / len(exp_crashes) * 100)
                reward_results[key].append(max(exp_rewards))
            else:
                crash_results[key] = [(sum(exp_crashes) / len(exp_crashes) * 100)]
                reward_results[key] = [(max(exp_rewards))]

    results = []
    for key in crash_results.keys():
        values = key.split("|")
        task_name = values[0]
        traj_type = values[1]
        safe_filter = values[2]
        unsafe_filter = values[3]

        mean_acc_reward = round(np.mean(reward_results[key]))
        std_acc_reward = round(np.std(reward_results[key]), 1)
        mean_crashes = round(np.mean(crash_results[key]))
        std_crashes = round(np.std(crash_results[key]), 1)

        row = [alg_name, task_name, traj_type, safe_filter, unsafe_filter, f"{mean_acc_reward}+{std_acc_reward}", f"{mean_crashes}+{std_crashes}"]
        results.append(row)

    df = pd.DataFrame(results, columns = ["Algorithm", "Task", "Traj. Type", "Safe Filter", "Unsafe Filter", "Acc. Reward", "Crashes"])
    df.to_csv(f'{alg_name}.csv')


def results():
    results_standard(alg_name = "SAC")
    results_DEFENDER(alg_name = "SAC_DEFENDER")
    results_standard(alg_name = "TD3")
    results_DEFENDER(alg_name = "TD3_DEFENDER")

if __name__ == '__main__':
    results()
