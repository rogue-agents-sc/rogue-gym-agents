from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
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

# define the environment
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

# define the model
config = DecisionTransformerConfig(
    state_dim=2048,
    act_dim=env.action_space.n,
    hidden_size=128,
    max_ep_len=1000,
    n_positions=2048,
    action_tanh=False,
)
model = DecisionTransformerModel(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# evaluation
device = torch.device("cpu")
model = model.to(device)
model.eval()

state = env.reset().__str__()
print(state)

tokens = tokenizer.encode(
    state,
    max_length=model.config.n_positions,
    padding="longest",
    truncation=True,
    add_special_tokens=True,
)

print(tokens)
print(len(tokens), len(state))

print(tokenizer.decode(tokens))
