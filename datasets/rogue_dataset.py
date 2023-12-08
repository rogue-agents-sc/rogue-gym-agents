import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json

class RogueDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        self.action_dim = 11

    def __getitem__(self, index):
        episode = self.episodes[index]
        states = [s.replace("\\n", "\n") for s in episode['states']]
        actions = F.one_hot(torch.tensor(episode['actions']), num_classes=self.action_dim)
        rewards = torch.tensor(episode['rewards']).unsqueeze(1)
        returns_to_go = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            returns_to_go[t] = rewards[t] + (0 if t==len(rewards)-1 else returns_to_go[t + 1])

        done = torch.tensor(episode['is_terminal'])
        timesteps = torch.arange(actions.shape[0])
        return states, actions, rewards, returns_to_go, timesteps, done

    def __len__(self):
        return len(self.episodes)
 
def get_dataloader(batch_size, shuffle=True):
    episodes = []
    # get current path
    path = os.path.dirname(os.path.realpath(__file__))
    episode_dir = path + '/episodes/episodes.json'
    with open(episode_dir, 'r') as f:
        episodes = json.load(f)
    dataset = RogueDataset(episodes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
