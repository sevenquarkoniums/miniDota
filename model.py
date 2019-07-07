import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        self.args = args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc4 = nn.Linear(args.hidden_size, num_outputs)
        # scale the initial weights.
#        self.fc4.weight.data.mul_(1)
#        self.fc4.bias.data.mul_(0)

    def forward(self, x):
        if self.args.activation == 'tanh':
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            mu = self.fc4(x)
        elif self.args.activation == 'relu':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            mu = self.fc4(x)
        elif self.args.activation == 'swish':
            x = self.fc1(x)
            x = x * torch.sigmoid(x)
            x = self.fc2(x)
            x = x * torch.sigmoid(x)
            x = self.fc3(x)
            x = x * torch.sigmoid(x)
            mu = self.fc4(x)
        else:
            raise ValueError
        if self.args.actionType == 'continuous':
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
            return mu, std, logstd # std is always 1.
        elif self.args.actionType == 'discrete':
            mu = (torch.tanh(mu) + 1) / 2 # each item is within (0, 1).
            return mu


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
#        self.fc3.weight.data.mul_(0.1)
#        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v
