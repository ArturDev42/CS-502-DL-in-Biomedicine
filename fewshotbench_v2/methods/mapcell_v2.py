import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
from methods.meta_template import MetaTemplate
import wandb

'''
In this version of the MapCell Model we train one common loss.
'''
class MapCell_v2(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MapCell_v2, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        # Create Siamese subnetwork
        self.subnetwork = self.create_subnetwork(self.feat_dim)

        # Compile model with contrastive loss and Adam optimizer
        self.optimizer_snn = optim.RMSprop(
            self.subnetwork.parameters(), 
            lr=0.0001, 
            alpha=0.99, 
            eps=1,
            weight_decay=0,
            momentum=0)

    def create_subnetwork(self, input_shape):
        model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 32),
            nn.ReLU()
        )

        return model


    def set_forward(self, x, is_feature=False):
        '''
        Function to set forward pass through the Siamese network - Difference to Proto: use of Siamese Neural Network as distance metric.

        Args:
            x: Input for our Neural Network.

        Returns:
            Tensor: Scores based on the euclidean distances between the prototypes and queries.

        '''
        z_support, z_query = self.parse_feature(x, is_feature)
 
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = self.set_forward_snn([z_query, z_proto])
        scores = -dists

        return scores

    def set_forward_loss(self, x):
        '''
        Computes the loss for training our entire Model.

        Args:
            x ([Tensor, Tensor]): Input for our Siamese Neural Network

        Returns:
            Tensor: Output combined loss

        '''
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        labels = torch.nn.functional.one_hot(y_query.to(torch.int64), num_classes=self.n_way)
        labels = Variable(labels)

        # parameter of the loss to experiment with
        margin = 1.0

        first_term = ((1.0-labels.float())*scores**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0.0),margin-scores)**2
        contrastive_loss = (first_term + second_term).mean()

        print("contrastive", contrastive_loss)

        loss_ce = self.loss_fn(scores, y_query)

        print("loss_ce", loss_ce)

        alpha = 0.5  # Adjust the weight of each loss
        total_loss = alpha * loss_ce + (1 - alpha) * contrastive_loss

        total_loss = contrastive_loss

        # total_loss.backward(retain_graph=True)
        # self.optimizer_snn.step()

        return total_loss


    def set_forward_snn(self, x):
        '''
        Function to set forward pass through the Siamese network.

        Args:
            x ([Tensor, Tensor]): Input pairs for our Siamese Neural Network, where x[0] is the first element, and x[1] is the second element.

        Returns:
            Tensor: Euclidean distances between the output embeddings of the Siamese network.
        '''

        output1, output2 = self.subnetwork(x[0]), self.subnetwork(x[1])
        euclidean_dist_embeddings = euclidean_dist(output1,output2)

        return euclidean_dist_embeddings

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2).sqrt()

