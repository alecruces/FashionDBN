"""This script implements a Restricted Boltzmann Machine.
The RBM can be trained with the contrastive divergence algorithm and can be embedded in a hierarchical
model in the DBN class."""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import sys

BATCH_SIZE = 64


class RBM(nn.Module):
    """This class defines all the functions needed for an BinaryRBN model
    where the visible and hidden units are both considered binary.
    """
    def __init__(self,
                 visible_units=256,
                 hidden_units=64,
                 k=2,
                 learning_rate=1e-5,
                 learning_rate_decay=False,
                 weight_decay=.0002,
                 initial_momentum=.5,
                 final_momentum=.9,
                 xavier_init=False,
                 increase_to_cd_k=False,
                 use_gpu=False):
        """
        Defines the model
        W:Weights shape (visible_units,hidden_units)
        c:hidden unit bias shape (hidden_units , )
        b : visible unit bias shape(visible_units ,)
        """
        super(RBM, self).__init__()
        self.desc = "RBM"

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.momentum = initial_momentum
        self.final_momentum = final_momentum
        self.xavier_init = xavier_init
        self.increase_to_cd_k = increase_to_cd_k
        self.use_gpu = use_gpu
        self.batch_size = 16

        # Initialization
        if not self.xavier_init:
            self.W = torch.randn(self.visible_units,
                                 self.hidden_units) * 0.1  # weights
        else:
            self.xavier_value = torch.sqrt(
                torch.FloatTensor(
                    [1.0 / (self.visible_units + self.hidden_units)]))
            self.W = -self.xavier_value + \
                torch.rand(self.visible_units, self.hidden_units) * (2 * self.xavier_value)
        self.h_bias = torch.zeros(self.hidden_units)  # hidden layer bias
        self.v_bias = torch.zeros(self.visible_units)  # visible layer bias

        self.v_bias_update = torch.zeros(self.visible_units)
        self.h_bias_update = torch.zeros(self.hidden_units)
        self.grad_update = torch.zeros(self.visible_units, self.hidden_units)

        if self.use_gpu:
            self.W = self.W.cuda()
            self.h_bias = self.h_bias.cuda()
            self.v_bias = self.v_bias.cuda()
            self.v_bias_update = self.v_bias_update.cuda()
            self.h_bias_update = self.h_bias_update.cuda()
            self.grad_update = self.grad_update.cuda()

    def to_hidden(self, X):
        """Converts the data in visible layer to hidden layer
        also does sampling
        X here is the visible probabilities

        :param X: torch tensor shape = (n_samples , n_features)
        :return -  X_prob - new hidden layer (probabilities)
                    sample_X_prob - Gibbs sampling of hidden (1 or 0) based
                                on the value
        """
        X_prob = torch.matmul(X, self.W)
        X_prob = torch.add(X_prob, self.h_bias)  # W.x + c
        X_prob = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob, sample_X_prob

    def to_visible(self, X):
        """reconstructs data from hidden layer
        also does sampling
        X here is the probabilities in the hidden layer

        :param X: 
        :returns: X_prob - the new reconstructed layers(probabilities)
                    sample_X_prob - sample of new layer(Gibbs Sampling)
        """
        # computing hidden activations and then converting into probabilities
        X_prob = torch.matmul(X, self.W.transpose(0, 1))
        X_prob = torch.add(X_prob, self.v_bias)
        X_prob = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob, sample_X_prob

    def sampling(self, prob):
        """Bernoulli sampling done based on probabilities s

        :param prob:
        """
        s = torch.distributions.Bernoulli(prob).sample()
        return s

    def reconstruction_error(self, data):
        """Computes the reconstruction error for the data
        handled by pytorch by loss functions

        :param data:
        """
        return self.contrastive_divergence(data, False)

    def reconstruct(self, X, n_gibbs):
        """This will reconstruct the sample with k steps of gibbs Sampling

        :param X: 
        :param n_gibbs:
        """
        v = X
        for i in range(n_gibbs):
            prob_h_, h = self.to_hidden(v)
            prob_v_, v = self.to_visible(prob_h_)
        return prob_v_, v

    def contrastive_divergence(self,
                               input_data,
                               training=True,
                               n_gibbs_sampling_steps=1,
                               lr=0.001):
        """Implementation of the contrastive divergence algorithm.

        :param input_data:
        :param training:  (Default value = True)
        :param n_gibbs_sampling_steps:  (Default value = 1)
        :param lr:  (Default value = 0.001)
        """
        # positive phase
        positive_hidden_probabilities, positive_hidden_act = self.to_hidden(
            input_data)

        # calculating W via positive side
        positive_associations = torch.matmul(input_data.t(),
                                             positive_hidden_probabilities)

        # negative phase
        hidden_activations = positive_hidden_act
        for i in range(n_gibbs_sampling_steps):
            visible_probabilities, _ = self.to_visible(hidden_activations)
            hidden_probabilities, hidden_activations = self.to_hidden(
                visible_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        # calculating W via negative side
        negative_associations = torch.matmul(
            negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        if training:
            batch_size = self.batch_size

            g = (positive_associations - negative_associations)
            self.grad_update = self.momentum * self.grad_update + lr * (g / batch_size - self.weight_decay * self.W)
            self.v_bias_update = self.momentum * self.v_bias_update + lr * torch.sum(
                input_data - negative_visible_probabilities,
                dim=0) / batch_size
            self.h_bias_update = self.momentum * self.h_bias_update + lr * torch.sum(
                positive_hidden_probabilities - negative_hidden_probabilities,
                dim=0) / batch_size

            self.W += self.grad_update
            self.v_bias += self.v_bias_update
            self.h_bias += self.h_bias_update

        # Compute reconstruction error
        error = torch.mean(
            torch.sum((input_data - negative_visible_probabilities)**2, dim=0))

        return error, torch.sum(torch.abs(self.grad_update))

    def forward(self, input_data):
        """Data to hidden.

        :param input_data:
        """
        return self.to_hidden(input_data)

    def step(self, input_data, epoch, num_epochs):
        """Includes the foward prop plus the gradient descent
            Use this for training

        :param input_data: 
        :param epoch: 
        :param num_epochs:
        """
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(
                math.ceil((epoch / num_epochs) * self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate

        if epoch > 5:
            self.momentum = self.final_momentum

        return self.contrastive_divergence(input_data, True,
                                           n_gibbs_sampling_steps, lr)

    def train(self, train_dataloader, num_epochs=50, batch_size=16):
        """Main training procedure.

        :param train_dataloader: 
        :param num_epochs:  (Default value = 50)
        :param batch_size:  (Default value = 16)
        """

        self.batch_size = batch_size
        if (isinstance(train_dataloader, torch.utils.data.DataLoader)):
            train_loader = train_dataloader
        else:
            train_loader = torch.utils.data.DataLoader(train_dataloader,
                                                       batch_size=batch_size)

        print("|Epoch |avg_rec_err |std_rec_err  |mean_grad |std_grad  |")
        for epoch in range(1, num_epochs + 1):
            epoch_err = 0.0
            n_batches = int(len(train_loader))
            # print(n_batches)

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            for i, (batch, _) in enumerate(train_loader):

                batch = batch.view(len(batch), self.visible_units)

                if self.use_gpu:
                    batch = batch.cuda()
                cost_[i - 1], grad_[i - 1] = self.step(batch, epoch,
                                                       num_epochs)

            if epoch % 10 == 0:
                print("|{:02d}    |{:.4f}     "
                      "|{:.4f}       |{:.4f}   "
                      "|{:.4f}     |".format(epoch, torch.mean(cost_), torch.std(cost_), torch.mean(grad_), torch.std(grad_)))

        return
