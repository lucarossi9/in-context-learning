import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """""
    The class implements a 1-hidden layer neural network
    """""

    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(NeuralNetwork,
              self).__init__()  # Return a proxy object that delegates method calls to a parent or sibling class of type
        # This is useful for accessing inherited methods that have been overridden in a class.
        # create sequential layers
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        """
        Implements forward method for a single NN.
        :param x: Torch tensor of inputs.
        :return: out: Torch tensor with the output of the NN.
        """
        out = self.net(x)  # forward method
        return out


class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):  # model_class_init_args is a keyword
        # argument (a dict that does not have to be specified)
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for i in range(num_models)]  # list of NN to run in parallel
        )  # self.net is a list of modules

    def forward(self, xs):
        """
        Implements forward method for parallel NN.
        :param xs: Torch tensor of inputs, one for each NN.
        :return: outs: a tensor, in the first dimension we concatenate the output of different NN.
        """
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out  # assign to the first dimension of each output the output of one network
        return outs
