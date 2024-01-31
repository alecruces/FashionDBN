"""This file contains the implementation of a Deep Belief Network, stacking several Restricted Boltzmann Machines
implemented in RBM.py."""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from RBM import RBM


class DBN(nn.Module):
    """Class implementing a DBN using the basic RBM class."""
    def __init__(self,
                 visible_units=256,
                 hidden_units=[64, 100],
                 k=2,
                 learning_rate=1e-5,
                 learning_rate_decay=False,
                 weight_decay=.0002,
                 initial_momentum=.5,
                 final_momentum=.9,
                 xavier_init=False,
                 increase_to_cd_k=False,
                 use_gpu=False):
        super(DBN, self).__init__()

        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.n_layers = len(hidden_units)
        self.rbm_layers = []
        self.rbm_nodes = []

        # Creating different RBM layers
        for i in range(self.n_layers):
            input_size = 0
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size,
                      hidden_units=hidden_units[i],
                      k=k,
                      learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      weight_decay=weight_decay,
                      initial_momentum=initial_momentum,
                      final_momentum=final_momentum,
                      xavier_init=xavier_init,
                      increase_to_cd_k=increase_to_cd_k,
                      use_gpu=use_gpu)

            self.rbm_layers.append(rbm)

        # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [
            nn.Parameter(self.rbm_layers[i].W.data.clone())
            for i in range(self.n_layers - 1)
        ]
        self.W_gen = [
            nn.Parameter(self.rbm_layers[i].W.data)
            for i in range(self.n_layers - 1)
        ]
        self.bias_rec = [
            nn.Parameter(self.rbm_layers[i].h_bias.data.clone())
            for i in range(self.n_layers - 1)
        ]
        self.bias_gen = [
            nn.Parameter(self.rbm_layers[i].v_bias.data)
            for i in range(self.n_layers - 1)
        ]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)

        for i in range(self.n_layers - 1):
            self.register_parameter('W_rec%i' % i, self.W_rec[i])
            self.register_parameter('W_gen%i' % i, self.W_gen[i])
            self.register_parameter('bias_rec%i' % i, self.bias_rec[i])
            self.register_parameter('bias_gen%i' % i, self.bias_gen[i])

    def forward(self, input_data):
        """running the forward pass
        do not confuse with training this just runs a forward pass

        :param input_data:
        """
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0], -1))  # flatten
            p_v, v = self.rbm_layers[i].to_hidden(v)
        return p_v, v

    def reconstruct(self, input_data):
        """go till the final layer and then reconstruct

        :param input_data:
        """
        p_h = input_data
        for i in range(len(self.rbm_layers)):
            p_h = p_h.view((p_h.shape[0], -1)).type(torch.FloatTensor).to(self.device)  # flatten
            p_h, h = self.rbm_layers[i].to_hidden(p_h)

        p_v = p_h
        for i in range(len(self.rbm_layers) - 1, -1, -1):
            p_v = p_v.view((p_v.shape[0], -1)).type(torch.FloatTensor).to(self.device)
            p_v, v = self.rbm_layers[i].to_visible(p_v)
        return p_v, v

    def train_static(self,
                     train_data,
                     train_labels,
                     num_epochs=50,
                     batch_size=10):
        """Greedy Layer By Layer training
        Keeping previous layers as static

        :param train_data: 
        :param train_labels: 
        :param num_epochs:  (Default value = 50)
        :param batch_size:  (Default value = 10)
        """
        tmp = train_data

        for i in range(len(self.rbm_layers)):
            print("-" * 20)
            print("Training RBM layer {}".format(i + 1))

            # transform to torch tensors
            tensor_x = tmp.type(torch.FloatTensor)
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(
                tensor_x, tensor_y)  # create your dataset
            _dataloader = torch.utils.data.DataLoader(
                _dataset, batch_size=batch_size,
                drop_last=True)  # create your DataLoader

            self.rbm_layers[i].train(_dataloader, num_epochs, batch_size)
            # print(train_data.shape)
            v = tmp.view((tmp.shape[0], -1)).type(torch.FloatTensor)  # flatten
            if self.rbm_layers[i].use_gpu:
                v = v.cuda()
            p_v, v = self.rbm_layers[i].forward(v)
            tmp = p_v
            # print(v.shape)
        return

    def train_ith(self, train_data, train_labels, num_epochs, batch_size,
                  ith_layer):
        """taking ith layer at once
        can be used for fine tuning

        :param train_data: 
        :param train_labels: 
        :param num_epochs: 
        :param batch_size: 
        :param ith_layer:
        """
        if (ith_layer - 1 > len(self.rbm_layers) or ith_layer <= 0):
            print("Layer index out of range")
            return
        ith_layer = ith_layer - 1
        v = train_data.view((train_data.shape[0], -1)).type(torch.FloatTensor)

        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].forward(v)

        tmp = v
        tensor_x = tmp.type(torch.FloatTensor)  # transform to torch tensors
        tensor_y = train_labels.type(torch.FloatTensor)
        _dataset = torch.utils.data.TensorDataset(
            tensor_x, tensor_y)  # create your datset
        _dataloader = torch.utils.data.DataLoader(_dataset,
                                                  batch_size=batch_size,
                                                  drop_last=True)
        self.rbm_layers[ith_layer].train(_dataloader, num_epochs, batch_size)
        return
