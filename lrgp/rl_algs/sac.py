import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .base import FFNetwork, Algorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
# Paper: https://arxiv.org/abs/1801.01290
# Note: Not the author's implementation


class Actor(nn.Module):
    """
    Uses tanh activation --> [-1, 1] symmetric output
    It scales the output using action_bound, and moves it using action_offset
    action = output * action_bound + action_offset
    """

    def __init__(self, state_dim, action_dim, hidden_dims, action_bound, action_offset, min_log_sigma=-20,
                 max_log_sigma=2, requires_grad=True):
        super().__init__()

        self.fc = FFNetwork(state_dim, hidden_dims[-1], hidden_dims[:-1], nn.ReLU(), requires_grad).to(device)
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_sigma_head = nn.Linear(hidden_dims[-1], action_dim)

        self.action_bound = torch.FloatTensor(action_bound).to(device)
        self.action_offset = torch.FloatTensor(action_offset).to(device)

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.epsilon = 1e-6

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)
        z = normal.rsample().to(device)
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1)
        action = action * self.action_bound + self.action_offset
        return action, log_prob

    def select_action(self, state, test):
        with torch.no_grad():
            x = self.fc(state)
            mu = self.mu_head(x)
            if not test:
                log_sigma = self.log_sigma_head(x)
                log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
                sigma = log_sigma.exp()
                normal = Normal(mu, sigma)
                mu = normal.rsample().to(device)
            action = torch.tanh(mu)
        return action * self.action_bound + self.action_offset


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims, requires_grad=True):
        super().__init__()

        self.network = FFNetwork(state_dim, 1, hidden_dims, None, requires_grad).to(device)

    def forward(self, state):
        return self.network(state)


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, requires_grad=True):
        super().__init__()

        self.network = FFNetwork(state_dim + action_dim, 1, hidden_dims, requires_grad=requires_grad).to(device)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.network(state_action)


class SAC(Algorithm):
    def __init__(
            self,
            state_dim,
            action_dim,

            action_bound,
            action_offset,

            gamma=0.99,
            tau=0.005,
            alpha=1,

            actor_hidden_dims=(128, 128, 128, 128),
            critic_hidden_dims=(128, 128, 128, 128),
            q_hidden_dims=(128, 128, 128, 128),
            actor_lr=3e-4,
            critic_lr=3e-4,
            q_lr=3e-4,
    ):
        super().__init__()

        # Init Actor network
        self.policy = Actor(state_dim, action_dim, actor_hidden_dims, action_bound, action_offset).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), actor_lr)

        # Init Critic networks
        self.value = Critic(state_dim, critic_hidden_dims).to(device)
        self.value_target = Critic(state_dim, critic_hidden_dims, False).to(device)
        self.copy_parameters(self.value, self.value_target)
        self.value_loss = nn.MSELoss()
        self.value_optimizer = optim.Adam(self.value.parameters(), critic_lr)

        # Init Q networks
        self.q_1 = Q(state_dim, action_dim, q_hidden_dims)
        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), q_lr)
        self.q_1_loss = nn.MSELoss()

        self.q_2 = Q(state_dim, action_dim, q_hidden_dims)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), q_lr)
        self.q_2_loss = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, goal, test):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.policy.select_action(state, test)
        return action.squeeze().cpu().numpy()

    def update(self, replay_buffer, batch_size):
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convert np arrays into tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        # Train Q networks
        q_value_1 = self.q_1(state, action).squeeze(-1)
        q_value_2 = self.q_2(state, action).squeeze(-1)
        next_v_value = self.value_target(next_state).squeeze(-1)

        q_value_target = reward + (1 - done) * self.gamma * next_v_value

        q_1_loss = self.q_1_loss(q_value_1, q_value_target)
        q_2_loss = self.q_2_loss(q_value_2, q_value_target)

        self.q_1_optimizer.zero_grad()
        q_1_loss.backward()
        self.q_1_optimizer.step()

        self.q_2_optimizer.zero_grad()
        q_2_loss.backward()
        self.q_2_optimizer.step()

        # Train Value network
        v_value = self.value(state).squeeze(-1)

        new_action, log_prob = self.policy(state)
        q_value = torch.min(self.q_1(state, new_action), self.q_2(state, new_action)).squeeze(-1)
        target_v_value = q_value - self.alpha * log_prob

        v_loss = self.value_loss(v_value, target_v_value.detach())  # We don't update q or policy network here

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # Train Actor network
        policy_loss = (self.alpha * log_prob - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.update_parameters_soft(self.value, self.value_target, self.tau)

    def save(self, path, job_name):
        torch.save(self.policy.state_dict(), os.path.join(path, f"{job_name}_actor.pth"))
        # torch.save(self.value.state_dict(), os.path.join(path, f"{job_name}_critic.pth"))
        # torch.save(self.q_1.state_dict(), os.path.join(path, f"{job_name}_q.pth"))

    def load(self, path, job_name):
        self.policy.load_state_dict(torch.load(os.path.join(path, f"{job_name}_actor.pth"), device))
        # self.value.load_state_dict(torch.load(os.path.join(path, f"{job_name}_critic.pth"), device))
        # self.q_1.load_state_dict(torch.load(os.path.join(path, f"{job_name}_q.pth"), device))

        # self.copy_parameters(self.q_1, self.q_2)
        # self.copy_parameters(self.critic, self.critic_target)


class SACStateGoal(Algorithm):
    """
    SAC for State-Goal Envs
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            goal_dim,

            action_bound,
            action_offset,

            gamma=0.99,
            tau=0.005,
            alpha=1,

            actor_hidden_dims=(128, 128, 128, 128),
            critic_hidden_dims=(128, 128, 128, 128),
            q_hidden_dims=(128, 128, 128, 128),
            actor_lr=3e-4,
            critic_lr=3e-4,
            q_lr=3e-4,
    ):
        super().__init__()

        # Init Actor network
        self.policy = Actor(state_dim + goal_dim, action_dim, actor_hidden_dims, action_bound, action_offset).to(device)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), actor_lr)

        # Init Critic networks
        self.value = Critic(state_dim + goal_dim, critic_hidden_dims).to(device)
        self.value_target = Critic(state_dim + goal_dim, critic_hidden_dims, False).to(device)
        self.copy_parameters(self.value, self.value_target)
        self.value_loss = nn.MSELoss()
        self.value_optimizer = optim.AdamW(self.value.parameters(), critic_lr)

        # Init Q networks
        self.q_1 = Q(state_dim + goal_dim, action_dim, q_hidden_dims)
        self.q_1_optimizer = optim.AdamW(self.q_1.parameters(), q_lr)
        self.q_1_loss = nn.MSELoss()

        self.q_2 = Q(state_dim + goal_dim, action_dim, q_hidden_dims)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), q_lr)
        self.q_2_loss = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, goal, test):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(device)
        state_goal = torch.cat([state, goal], dim=-1)
        action = self.policy.select_action(state_goal, test)
        return action.squeeze().cpu().numpy()

    def update(self, replay_buffer, batch_size):
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, goal, done = replay_buffer.sample(batch_size)

        # Convert np arrays into tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        done = torch.FloatTensor(done).to(device)

        # Create state-goals
        state = torch.cat([state, goal], dim=-1)
        next_state = torch.cat([next_state, goal], dim=-1)

        # Train Q networks
        q_value_1 = self.q_1(state, action).squeeze(-1)
        q_value_2 = self.q_2(state, action).squeeze(-1)
        next_v_value = self.value_target(next_state).squeeze(-1)

        q_value_target = reward + (1 - done) * self.gamma * next_v_value

        q_1_loss = self.q_1_loss(q_value_1, q_value_target)
        q_2_loss = self.q_2_loss(q_value_2, q_value_target)

        self.q_1_optimizer.zero_grad()
        q_1_loss.backward()
        self.q_1_optimizer.step()

        self.q_2_optimizer.zero_grad()
        q_2_loss.backward()
        self.q_2_optimizer.step()

        # Train Value network
        v_value = self.value(state).squeeze(-1)

        new_action, log_prob = self.policy(state)
        q_value = torch.min(self.q_1(state, new_action), self.q_2(state, new_action)).squeeze(-1)
        target_v_value = q_value - self.alpha * log_prob

        v_loss = self.value_loss(v_value, target_v_value.detach())  # We don't update q or policy network here

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # Train Actor network
        policy_loss = (self.alpha * log_prob - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.update_parameters_soft(self.value, self.value_target, self.tau)

    def save(self, path, job_name):
        torch.save(self.policy.state_dict(), os.path.join(path, f"{job_name}_actor.pth"))
        # torch.save(self.value.state_dict(), os.path.join(path, f"{job_name}_critic.pth"))
        # torch.save(self.q_1.state_dict(), os.path.join(path, f"{job_name}_q.pth"))

    def load(self, path, job_name):
        self.policy.load_state_dict(torch.load(os.path.join(path, f"{job_name}_actor.pth"), device))
        # self.value.load_state_dict(torch.load(os.path.join(path, f"{job_name}_critic.pth"), device))
        # self.q_1.load_state_dict(torch.load(os.path.join(path, f"{job_name}_q.pth"), device))

        # self.copy_parameters(self.q_1, self.q_2)
        # self.copy_parameters(self.critic, self.critic_target)
