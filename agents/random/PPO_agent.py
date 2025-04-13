import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==== PPO Policy Network ====
class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)  # raw logits for discrete actions
        return logits

# ==== 初始化 PPO 模型 ====
# 假设观测维度和动作维度是固定的（建议你根据实际环境填写）
OBS_DIM = 20   # example
ACT_DIM = 5    # example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = PPOPolicy(OBS_DIM, ACT_DIM).to(device)
policy_net.load_state_dict(torch.load("ppo_model.pth", map_location=device))
policy_net.eval()

# ==== 主控制函数 ====
def my_controller(observation, action_space, is_act_continuous=False):
    action_list = []

    for obs in observation:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # [1, obs_dim]
        with torch.no_grad():
            logits = policy_net(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()  # 或使用采样: torch.multinomial(probs, 1)

        # 转为 one-hot 编码
        act = [0] * ACT_DIM
        act[action_idx] = 1
        action_list.append(act)

    return action_list