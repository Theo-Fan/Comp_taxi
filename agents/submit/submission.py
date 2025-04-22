import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width=64, max_action=1.0, is_continuous=True):
        super(Actor, self).__init__()
        self.is_continuous = is_continuous
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean = nn.Linear(hidden_width, action_dim)
        if self.is_continuous:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        if self.is_continuous:
            mean = self.max_action * torch.tanh(self.mean(a))
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
        else:
            logits = self.mean(a)
            dist = Categorical(logits=logits)
        return dist

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width=64):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_width=64, max_action=1.0, is_continuous=True):
        self.actor = Actor(state_dim, action_dim, hidden_width, max_action, is_continuous)
        self.critic = Critic(state_dim, hidden_width)
        self.is_continuous = is_continuous

        
        # PPO超参数
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps = 0.2
        self.K_epochs = 10
        
        # 经验池
        self.memory = []
        self.batch_size = 64
        
        # 训练统计
        self.episode_rewards = []
        self.best_reward = float('-inf')

        self.critic_net_path = os.path.dirname(os.path.abspath(__file__)) + "/final_critic.pth"
        self.actor_net_path = os.path.dirname(os.path.abspath(__file__)) + "/final_actor.pth"
        
        self.actor.load_state_dict(torch.load(self.actor_net_path))
        self.critic.load_state_dict(torch.load(self.critic_net_path))

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            if self.is_continuous:
                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state)
                return action.numpy(), action_log_prob.numpy(), value.numpy()
            else:
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                value = self.critic(state)
                return action.numpy(), action_log_prob.numpy(), value.numpy()

ppo_agent = None
current_episode = 0
episode_reward = 0
last_state = None
last_action = None
last_log_prob = None
last_value = None

def my_controller(observation, action_space, is_act_continuous=False):
    global ppo_agent, current_episode, episode_reward, last_state, last_action, last_log_prob, last_value
    
    if ppo_agent is None:
        obs = observation['obs']['raw_obs']
        state_dim = len(obs) if isinstance(obs, np.ndarray) else len(obs[0])
        action_dim = action_space[0].shape[0] if is_act_continuous else action_space[0].n
        ppo_agent = PPOAgent(state_dim, action_dim, is_continuous=is_act_continuous)
    

    state = observation['obs']['raw_obs']
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    

    if last_state is None:
        current_episode += 1
        episode_reward = 0
        print(f"Starting episode {current_episode}") 
    

    action, log_prob, value = ppo_agent.select_action(state)
    
    # 如果是连续动作空间，需要将动作限制在[-1, 1]范围内
    if is_act_continuous:
        action = np.clip(action, -1.0, 1.0)
        action = action.squeeze()  
    
    if last_state is not None:
        reward = observation.get('reward', 0)
        episode_reward += reward
        done = observation.get('done', False)
        ppo_agent.memory.append((last_state, last_action, last_log_prob, reward, state, done))
        

        if done or len(ppo_agent.memory) >= ppo_agent.batch_size:
            # ppo_agent.update()
            ppo_agent.episode_rewards.append(episode_reward)

    
    last_state = state
    last_action = action
    last_log_prob = log_prob
    last_value = value
    
    return [action]