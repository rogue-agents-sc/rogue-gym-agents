from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
from rogue_gym.envs import (
    ImageSetting,
    RogueEnv,
    StatusFlag,
    StairRewardEnv,
    PlayerState,
)
from typing import Tuple, Union

TARGET_RETURN = 100

ENV_CONFIG = {
    "seed": 100,
    "hide_dungeon": False,
    "enemies": {
        "enemies": [],
    },
}

EXPAND = ImageSetting(status=StatusFlag.DUNGEON_LEVEL)


class SecondFloorEnv(StairRewardEnv):
    def step(self, action: Union[int, str]) -> Tuple[PlayerState, float, bool, None]:
        state, reward, end, info = super().step(action)
        if self.current_level == 3:
            end = True
        return state, reward, end, info

    def __repr__(self):
        return super().__repr__()


env = SecondFloorEnv(RogueEnv(config_dict=ENV_CONFIG, image_setting=EXPAND), 100.0)
print(env.observation_space, env.action_space)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class HRLPolicy(torch.nn.Module):
    def __init__(self, action_space, state_dim, hidden_size, num_goals):
        super(HRLPolicy, self).__init__()
        self.action_space = action_space
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_goals = num_goals

        self.embedding = torch.nn.Embedding(num_goals, hidden_size)
        self.policy = torch.nn.Linear(state_dim + hidden_size, action_space)

    def forward(self, state, goal):
        goal_embedding = self.embedding(goal)
        state_goal_concat = torch.cat((state, goal_embedding), dim=1)
        action_logits = self.policy(state_goal_concat)
        return action_logits


class HRLModel(torch.nn.Module):
    def __init__(self, state_dim, action_space, num_goals, hidden_size):
        super(HRLModel, self).__init__()
        self.state_dim = state_dim
        self.action_space = action_space
        self.num_goals = num_goals
        self.hidden_size = hidden_size

        self.low_level_policy = HRLPolicy(action_space, state_dim, hidden_size, num_goals)
        self.goal_decoder = torch.nn.Linear(hidden_size, num_goals)

    def forward(self, state):
        
        goal_logits = self.goal_decoder(state)
        goal_probabilities = torch.nn.functional.softmax(goal_logits, dim=1)
        sampled_goal = torch.multinomial(goal_probabilities, 1)

        action_logits = self.low_level_policy(state, sampled_goal)
        return action_logits, sampled_goal

hrl_model = HRLModel(state_dim=2048, action_space=env.action_space.n, num_goals=3, hidden_size=128)

device = torch.device("cpu")
hrl_model = hrl_model.to(device)
hrl_model.eval()

state = env.reset().__str__()
tokens = tokenizer.encode(state)
states = torch.tensor([tokens]).to(device=device, dtype=torch.float32)

with torch.no_grad():
    action_logits, sampled_goal = hrl_model(states)
    action = action_logits.argmax(dim=1).item()
    print(f"Selected Action: {env.ACTIONS[action]}, Sampled Goal: {sampled_goal.item()}")

#don't need to encode the state, just pass in the state with the model. The state of the rogue gym comes in as a 3x3 placement with "coordinates" already