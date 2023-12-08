import sys
import os

# add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
    AutoTokenizer,
)
import torch
from rogue_gym.envs import (
    ImageSetting,
    DungeonType,
    RogueEnv,
    StatusFlag,
    StairRewardEnv,
)
from rogue_gym.rainy_impls import RogueEnvExt
from datasets.rogue_dataset import get_dataloader

# define the environment
TARGET_RETURN = 200.0

ENV_CONFIG = {
    "width": 32,
    "height": 16,
    "seed_range": [0, 40],
    "hide_dungeon": True,
    "dungeon": {
        "style": "rogue",
        "room_num_x": 2,
        "room_num_y": 2,
    },
    "enemies": {
        "enemies": [],
    },
}

EXPAND = ImageSetting(dungeon=DungeonType.SYMBOL, status=StatusFlag.EMPTY)

env = RogueEnvExt(StairRewardEnv(
    RogueEnv(
        config_dict=ENV_CONFIG,
        max_steps=500,
        stair_reward=50.0,
        image_setting=EXPAND
    ),
    100.0
))
screen_size = env.unwrapped.screen_size()
s_dim = screen_size[0] * screen_size[1]

# define the model
device = torch.device("cpu")
config = DecisionTransformerConfig(
    state_dim=s_dim,
    act_dim=env.action_space.n,
    hidden_size=128,
    max_ep_len=1000,
    n_positions=2048,
    action_tanh=False,
    vocab_size=50257,
)
model = DecisionTransformerModel(config)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# training loop
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 5
num_epochs = 1

print("=== Starting training loop ===")
loss = None
for epoch in range(num_epochs):
    print(f"=== Epoch {epoch} ===")
    for state, actions, rewards, returns_to_go, timesteps, done in get_dataloader(batch_size=batch_size):
        batch_size, ep_len = actions.shape[0], actions.shape[1]
        tokens = []
        for i in range(ep_len):
            tmp = []
            for j in range(batch_size):
                tmp.append(tokenizer.encode(
                    state[i][j],
                    max_length=s_dim,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                ))
            tokens.append(tmp)
        
        states = torch.tensor(tokens).permute(1, 0, 2).to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)
        rewards = rewards.to(device=device, dtype=torch.float32)
        returns_to_go = returns_to_go.to(device=device, dtype=torch.float32)
        timesteps = timesteps.to(device=device, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, ep_len, device=device, dtype=torch.float32)
    
        state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        # permute the tensors since CrossEntropyLoss expects (N, C, L)
        action_preds = action_preds.permute(0, 2, 1)
        actions = actions.permute(0, 2, 1)

        pa = env.unwrapped.ACTIONS[action_preds.argmax(dim=1)[0][0].item()]
        a = env.unwrapped.ACTIONS[actions.argmax(dim=1)[1][0].item()]

        # compute cross entropy loss
        loss = loss_fn(action_preds, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss:", loss.item(), " -> sample action:", a, "predicted action:", pa)

    # save model weights
    print("Saving model weights")
    torch.save(model.state_dict(), 'model_weights.pth')

# evaluation
print("Starting evaluation...")
model.eval()

state = env.reset().__str__()
tokens = tokenizer.encode(
    state,
    max_length=s_dim,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
)

# (batch_size, episode_length, state_dim)
states = torch.tensor([tokens]).reshape(1, 1, -1).to(device=device, dtype=torch.float32)
# (batch_size, episode_length, action_dim)
actions = torch.zeros((1, 1, model.config.act_dim), device=device, dtype=torch.float32)
# (batch_size, episode_length, 1)
rewards = torch.zeros((1, 1, 1), device=device, dtype=torch.float32)
# (batch_size, episode_length, 1)
target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1, 1)
# (batch_size, episode_length)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# (batch_size, episode_length)
attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

done = False
ts = 0

# forward pass
print(state.__str__())
with torch.no_grad():
    while not done and ts < 200:
        state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        # get the max index of action_preds
        action_idx = action_preds.argmax(dim=2)[:, -1]
        action = env.unwrapped.ACTIONS[action_idx]
        print(ts, "- action:", action_idx, action, env.unwrapped.ACTION_MEANINGS[action], " reward:", reward, )

        state, reward, done, _ = env.step(action)

        tokens = tokenizer.encode(
            state.__str__(),
            max_length=s_dim,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        states = torch.cat(
            (
                states,
                torch.tensor([tokens])
                .reshape(1, 1, -1)
                .to(device=device, dtype=torch.float32),
            ),
            1,
        )
        actions = torch.cat(
            (
                actions,
                action_preds[:, -1, :].unsqueeze(1),
            ),
            1,
        )
        rewards = torch.cat(
            (rewards, torch.tensor(reward, device=device).reshape(1, 1, 1)), 1
        )
        target_return = torch.cat(
            (target_return, torch.sub(target_return[0, -1], reward).reshape(1, 1, 1)), 1
        )
        attention_mask = torch.cat(
            (attention_mask, torch.zeros(1, 1, device=device, dtype=torch.float32)), 1
        )
        timesteps = torch.cat(
            (timesteps, torch.tensor(ts, device=device).reshape(1, 1)), 1
        )
        
        ts += 1

    print(state.__str__())
    # print(state.symbol_image_with_hist())
    env.unwrapped.save_actions("actions.json")
