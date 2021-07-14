import os
import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC

from .base import FFNetwork, Algorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Reinforcement Learning with Double Q-learning (DDQN)
# Paper: https://arxiv.org/abs/1509.06461
# Note: Not the author's implementation


class DDQNBase(Algorithm, ABC):
    def __init__(
            self,
            state_dim,
            action_dim,

            gamma=0.99,
            tau=0.005,

            hidden_dims=(128, 128, 128, 128),
            lr=3e-4,
    ):

        super().__init__()

        # Init networks
        self.value_network = FFNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = FFNetwork(state_dim, action_dim, hidden_dims, requires_grad=False).to(device)
        self.copy_parameters(self.value_network, self.target_network)
        self.mseLoss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.value_network.parameters(), lr)

        self.gamma = gamma
        self.tau = tau

    def save(self, path, job_name):
        torch.save(self.value_network.state_dict(), os.path.join(path, f"{job_name}.pth"))

    def load(self, path, job_name):
        self.value_network.load_state_dict(torch.load(os.path.join(path, f"{job_name}.pth"), device))
        # self.copy_parameters(self.value_network, self.target_network)


class DDQN(DDQNBase):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super().__init__(state_dim, action_dim, *args, **kwargs)

    def select_action(self, state):
        # Add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.value_network(state).argmax().item()

    def update(self, replay_buffer, batch_size):

        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convert np arrays into tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        # q_values
        q_value = self.value_network(state)
        next_q_value = self.value_network(next_state)
        next_q_value_target = self.target_network(next_state)

        # expected q_value using double network
        next_q_value = next_q_value_target.gather(1, torch.argmax(next_q_value, 1, keepdim=True)).squeeze(-1)
        expected_q_value = reward + (1 - done) * self.gamma * next_q_value

        # q_value for taken action
        q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(-1)

        # Optimize
        loss = self.mseLoss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_parameters_soft(self.value_network, self.target_network, self.tau)


class DDQNStateGoal(DDQNBase):
    """
    DDQN Algorithm for State-Goal Envs
    """
    def __init__(self, state_dim, action_dim, goal_dim, *args, **kwargs):
        super().__init__(state_dim + goal_dim, action_dim, *args, **kwargs)

    def select_action(self, state, goal):
        # Add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(device)
        state_goal = torch.cat([state, goal], dim=-1)
        with torch.no_grad():
            return self.value_network(state_goal).argmax().item()

    def update(self, replay_buffer, batch_size):

        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, goal, done = replay_buffer.sample(batch_size)

        # Convert np arrays into tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        done = torch.FloatTensor(done).to(device)

        # Create state-goals
        state = torch.cat([state, goal], dim=-1)
        next_state = torch.cat([next_state, goal], dim=-1)

        # q_values
        q_value = self.value_network(state)
        next_q_value = self.value_network(next_state)
        next_q_value_target = self.target_network(next_state)

        # expected q_value using double network
        next_q_value = next_q_value_target.gather(1, torch.argmax(next_q_value, 1, keepdim=True)).squeeze(-1)
        expected_q_value = reward + (1 - done) * self.gamma * next_q_value

        # q_value for taken action
        q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(-1)

        # Optimize
        loss = self.mseLoss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_parameters_soft(self.value_network, self.target_network, self.tau)
