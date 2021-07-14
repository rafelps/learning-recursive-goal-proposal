import torch.nn as nn


class FFNetwork(nn.Module):
    """
    Feed-Forward Network
    """
    def __init__(self, input_dim, output_dim, hidden_dims, output_activation=None, requires_grad=True):
        super().__init__()

        # Add input layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

        # Add hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Add output activation
        if output_activation is not None:
            layers.append(output_activation)

        self.network = nn.Sequential(*layers)

        if not requires_grad:
            for param in self.network.parameters():
                param.requires_grad = False

    def forward(self, state):
        return self.network(state)


class Algorithm:
    def __init__(self):
        pass

    @staticmethod
    def copy_parameters(from_network, to_network):
        to_network.load_state_dict(from_network.state_dict())

    @staticmethod
    def update_parameters_soft(from_network, to_network, tau):
        from_dict = from_network.state_dict()
        to_dict = to_network.state_dict()
        for k in from_dict:
            to_dict[k] = (1 - tau) * to_dict[k] + tau * from_dict[k]
        to_network.load_state_dict(to_dict)

    def select_action(self, *args):
        raise NotImplementedError

    def update(self, buffer, batch_size):
        raise NotImplementedError

    def load(self, path, jobname):
        raise NotImplementedError

    def save(self, path, jobname):
        raise NotImplementedError
