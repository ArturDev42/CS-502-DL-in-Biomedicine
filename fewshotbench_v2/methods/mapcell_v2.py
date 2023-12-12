import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
from methods.meta_template import MetaTemplate
import wandb


class MapCell_v2(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MapCell_v2, self).__init__(backbone, n_way, n_support)
        print("ProtoNet Backbone:", backbone)
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
        """
        Function to set forward pass for training the model.
        """
        z_support, z_query = self.parse_feature(x, is_feature)
 
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = self.set_forward_snn([z_query, z_proto])
        scores = -dists

        return scores

    # train loop
    def train_loop(self, epoch, train_loader, optimizer):
        """
        Function to train whole Model.
        """
        print_freq = 10

        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})

    def set_forward_loss(self, x):
        """
        Function to compute the overall forward loss.
        """
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        print("y_query", y_query)
        print("scores", scores)
        print("y_query", y_query.size())
        print("scores", scores.size())

        labels = torch.zeros(self.n_query * self.n_way, dtype=torch.float32).cuda()

        for i in range(self.n_way):
            labels[i * self.n_query : (i + 1) * n_query] = i

        labels = Variable(labels)

        print("labels", labels)
        print("labels", labels.size())

        # parameter of the loss to experiment with
        margin = 0.5

        first_term = ((1.0-labels.float())*scores**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0.0),margin-scores)**2
        contrastive_loss = (first_term + second_term).mean()

        loss_ce = self.loss_fn(scores, y_query)

        alpha = 0.5  # Adjust the weight of each loss
        total_loss = alpha * loss_ce + (1 - alpha) * loss_contrastive

        total_loss.backward()
        self.optimizer_snn.step()

        return total_loss


    def set_forward_snn(self, x):
        """
        Function to set forward pass through the Siamese network.
        """

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

