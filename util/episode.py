import json

class EpisodeHistory:
    def __init__(self):
        self.accumulated_rewards = []
        self.episode_lengths = []
        self.terminateds = []
        self.truncateds = []
        self.infos = []
        self.filtereds = []
        self.crashes = []

    def store_episode(self, accumulate_reward : float, episode_length : int, terminated : bool, truncated : bool, last_info : dict, crash : int, filtered : bool = False):
        self.accumulated_rewards.append(accumulate_reward)
        self.episode_lengths.append(episode_length)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.infos.append(last_info)
        self.filtereds.append(int(filtered))
        self.crashes.append(crash)

    def save_result(self, result_path : str):
        results = {
            "rewards" : self.accumulated_rewards,
            "lengths" : self.episode_lengths,
            "terminateds" : self.terminateds,
            "truncateds" : self.truncateds,
            "infos" : self.infos,
            "filtereds" : self.filtereds,
            "crashes" : self.crashes
        }
        with open(result_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)