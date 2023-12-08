from transformers import TrajectoryTransformerModel
import torch

model = TrajectoryTransformerModel.from_pretrained(
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
)
model.to(device)
model.eval()

observations_dim, action_dim, batch_size = 17, 6, 256
seq_length = observations_dim + action_dim + 1

trajectories = torch.LongTensor(
    [np.random.permutation(self.seq_length) for _ in range(batch_size)]
).to(device)
targets = torch.LongTensor(
    [np.random.permutation(self.seq_length) for _ in range(batch_size)]
).to(device)

outputs = model(
    trajectories,
    targets=targets,
    use_cache=True,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)
