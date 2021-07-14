import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.max_size = int(max_size)
        self.buffer = []
        self.position = 0

    def add(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class HERTransitionCreator:
    class Transition:
        # Pythonic view of a transition
        def __init__(self, state, action, reward, next_state, goal, done):
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state
            self.goal = goal
            self.done = done

        def to_tup(self):
            return self.state, self.action, self.reward, self.next_state, self.goal, self.done

    def __init__(self, state_goal_mapper=None):
        # Container for episode transitions
        self.original_transitions = list()

        # sgm to convert states into goals for hindsight
        if state_goal_mapper is None:
            state_goal_mapper = lambda x: x
        self.state_goal_mapper = state_goal_mapper

    def __len__(self):
        return len(self.original_transitions)

    def add(self, state, action, reward, next_state, goal, done):
        self.original_transitions.append(self.Transition(state, action, reward, next_state, goal, done))

    def create_and_insert(self, replay_buffer):
        """
        Apply 'future' HER --> Use a future next state as goal
        Add HER transitions to the buffer
        """
        for i, t in enumerate(self.original_transitions):
            # Select a future transition in the episode
            future = np.random.randint(i, len(self))
            # Take its next_state
            next_state = self.original_transitions[future].next_state
            # Convert into goal-space and set as goal for current transition
            t.goal = self.state_goal_mapper(next_state)
            # Compute binary reward --> If new goal was achieved in that transition 0, else -1
            t.reward = int(np.array_equal(self.state_goal_mapper(t.next_state), t.goal)) - 1

            # Add to replay buffer (original transitions were already added)
            replay_buffer.add(*t.to_tup())

        # Flush list for following episode
        self.original_transitions = list()
