
from stable_baselines3.sac.policies import MultiInputPolicy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import torch.nn.functional as F


class CustomAttentionPolicy(MultiInputPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         features_extractor_class=AttentionFeatureExtractor,
                         features_extractor_kwargs=dict(num_elements=10, input_dim=16),
                         **kwargs)


OBS_DIMS = 256

class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_elements=4, input_dim=OBS_DIMS, hidden_dim=64):
        super().__init__(observation_space, features_dim)

        self.attention = SetAttentionLayer(input_dim=input_dim, hidden_dim=hidden_dim)
        # self.fc = nn.Linear(num_elements * hidden_dim, features_dim)
        self.fc = nn.Linear(1024, features_dim)

    def forward(self, observations):
        # observations: [batch_size, num_elements * input_dim]
        batch_size = observations['observation'].shape[0]

        obs = observations['observation'].view(batch_size, 16, 16)  # reshape to [batch, N, D]

        attended, _ = self.attention(obs)             # [batch, N, hidden_dim]
        flat = attended.view(batch_size, -1)          # flatten all attended features
        return self.fc(flat)
    

class SetAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(weights, V)
        return attended, weights