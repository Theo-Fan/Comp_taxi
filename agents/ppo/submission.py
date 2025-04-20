import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os


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
        else:
            self.mean = nn.Linear(hidden_width, action_dim)

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
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
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
        
        # 加载预训练模型（如果有的话）
        try:
            self.actor.load_state_dict(torch.load("agents/ppo/final_actor.pth"))
            self.critic.load_state_dict(torch.load("agents/ppo/final_critic.pth"))
            print("Loaded pretrained models")
        except:
            print("No pretrained models found. Starting from scratch.")

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

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 将经验转换为张量
        states = torch.FloatTensor(np.array([m[0] for m in self.memory]))
        actions = torch.FloatTensor(np.array([m[1] for m in self.memory]))
        old_log_probs = torch.FloatTensor(np.array([m[2] for m in self.memory]))
        rewards = torch.FloatTensor(np.array([m[3] for m in self.memory]))
        next_states = torch.FloatTensor(np.array([m[4] for m in self.memory]))
        dones = torch.FloatTensor(np.array([m[5] for m in self.memory]))
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            td_target = rewards + self.gamma * next_values * (1 - dones)
            delta = td_target - values
            advantage = torch.zeros_like(delta)
            advantage[-1] = delta[-1]
            for t in reversed(range(len(delta)-1)):
                advantage[t] = delta[t] + self.gamma * self.lmbda * advantage[t+1]
        
        # PPO更新
        for _ in range(self.K_epochs):
            # 计算新的动作概率
            dist = self.actor(states)
            if self.is_continuous:
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
                new_log_probs = dist.log_prob(actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO目标
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新critic
            critic_loss = F.mse_loss(self.critic(states), td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # 清空经验池
        self.memory = []

    def save_models(self, prefix=""):
        """保存模型的方法"""
        if prefix:
            prefix = prefix + "_"
        torch.save(self.actor.state_dict(), f"agents/ppo/{prefix}actor.pth")
        torch.save(self.critic.state_dict(), f"agents/ppo/{prefix}critic.pth")
        print(f"Models saved with prefix: {prefix}")

# 全局变量
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
    
    # 处理观察值
    state = observation['obs']['raw_obs']
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    
    # 如果是新的回合
    if last_state is None:
        current_episode += 1
        episode_reward = 0
        print(f"Starting episode {current_episode}")  # 添加回合开始提示
    
    # 选择动作
    action, log_prob, value = ppo_agent.select_action(state)
    
    # 如果是连续动作空间，需要将动作限制在[-1, 1]范围内
    if is_act_continuous:
        action = np.clip(action, -1.0, 1.0)
        # 确保输出格式与random agent一致
        action = action.squeeze()  # 移除多余的维度
    
    if last_state is not None:
        reward = observation.get('reward', 0)
        episode_reward += reward
        done = observation.get('done', False)
        ppo_agent.memory.append((last_state, last_action, last_log_prob, reward, state, done))
        
        # 如果回合结束或经验池满了，更新网络
        if done or len(ppo_agent.memory) >= ppo_agent.batch_size:
            # ppo_agent.update()
            ppo_agent.episode_rewards.append(episode_reward)

    
    # 更新状态
    last_state = state
    last_action = action
    last_log_prob = log_prob
    last_value = value
    
    # 确保输出格式与random agent一致
    return [action]