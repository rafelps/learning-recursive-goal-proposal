import numpy as np
from gym_simple_minigrid.minigrid import SimpleMiniGridEnv

from .rl_algs.sac import SACStateGoal
from .utils.utils import ReplayBuffer


class HighPolicy:
    def __init__(self, env: SimpleMiniGridEnv, gamma: float = 1., tau: float = 0.005, br_size: int = 1e6):
        self.env = env

        state_shape = env.observation_space.shape[0]
        goal_shape = env.state_goal_mapper(env.observation_space.sample()).shape[0]
        # High agent proposes goals --> action space === goal space
        action_shape = goal_shape

        # Compute action bounds to convert SAC's action in the correct range
        action_low = env.state_goal_mapper(env.observation_space.low)
        action_high = env.state_goal_mapper(env.observation_space.high)
        action_high_corrected = 1 + action_high  # To adapt discrete env into SAC (continuous actions)

        action_bound = (action_high_corrected - action_low) / 2
        action_offset = (action_high_corrected + action_low) / 2

        # Init SAC algorithm, base learner for high agent
        self.alg = SACStateGoal(state_shape, action_shape, goal_shape, action_bound, action_offset, gamma, tau)

        self.clip_low = action_low
        self.clip_high = action_high

        self.replay_buffer = ReplayBuffer(br_size)

        self.episode_runs = list()

    def select_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # SAC action is continuous [low, high + 1]
        action = self.alg.select_action(state, goal, False)
        # Discretize using floor --> discrete [low, high + 1]
        action = np.floor(action)
        # In case action was exactly high + 1, it is out of bounds. Clip
        action = np.clip(action, self.clip_low, self.clip_high)

        return action.astype(np.int)

    def select_action_test(self, state: np.ndarray, goal: np.ndarray, add_noise: bool = False) -> np.ndarray:
        action = self.alg.select_action(state, goal, True)

        noise = 0
        if add_noise:
            # Add small noise so we choose an adjacent goal position
            noise = np.random.randint(-1, 2, action.shape)
        action = action + noise

        # Discretize and clip
        action = np.floor(action)
        action = np.clip(action, self.clip_low, self.clip_high)

        return action.astype(np.int)

    def add_run_info(self, info: tuple):
        self.episode_runs.append(info)

    def add_penalization(self, transition: tuple):
        self.replay_buffer.add(*transition)

    def on_episode_end(self):
        # Create MonteCarlo-based transitions from episode runs
        # Hindsight goals --> Next state as proposed goal (as if low level acts optimally)

        # We do not create transitions between states and goals that are reachable within one run.
        # We do not need to learn them because no subgoals are required

        for i, (state_1, _, next_state_1) in enumerate(self.episode_runs):
            for j, (_, _, next_state_3) in enumerate(self.episode_runs[i + 1:], i + 1):
                # Used as final goal
                hindsight_goal_3 = self.env.state_goal_mapper(next_state_3)
                for k, (_, _, next_state_2) in enumerate(self.episode_runs[i:j], i):
                    # Used as intermediate goal or proposed action
                    hindsight_goal_2 = self.env.state_goal_mapper(next_state_2)
                    self.replay_buffer.add(state_1,             # state
                                           hindsight_goal_2,    # action <-> proposed goal
                                           -(j - i + 1),        # reward <-> - N runs
                                           next_state_1,        # (NOT USED) next_state
                                           hindsight_goal_3,    # goal
                                           True)                # done --> Q-value = Reward (no bootstrap / Bellman eq)
        self.episode_runs = list()

    def update(self, n_updates: int, batch_size: int):
        if len(self.replay_buffer) > 0:
            for _ in range(n_updates):
                self.alg.update(self.replay_buffer, batch_size)

    def save(self, path: str):
        self.alg.save(path, "high")

    def load(self, path: str):
        self.alg.load(path, "high")
