import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
from methods.meta_template import MetaTemplate
import wandb


class MapCell(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MapCell, self).__init__(backbone, n_way, n_support)
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

    # intuitiv training
    def train_siamese_network(self, x, y):
        """
        Function for training the Siamese network. Creates pairs of samples and updates weights by minimizing contrastive loss.
        """
        self.subnetwork.train()
        freq = 10

        x, _ = self.parse_feature(x, is_feature=False)

        print(x.size())

        y = y[:self.n_support].cpu().data.numpy()
        pairs, pair_labels = self.split_data_into_pairs(x, y)

        # Convert pairs and labels to PyTorch variables
        pairs = [Variable(pair.to(self.device)) for pair in pairs]
        pair_labels = Variable(pair_labels.to(self.device))

        self.optimizer_snn.zero_grad()

        # Forward pass for Siamese network training
        i = 1
        for pair0, pair1, label in zip(pairs[0], pairs[1], pair_labels):
            snn_loss = self.set_forward_snn_loss([pair0, pair1], label)
            if i % freq == 0:
                snn_loss.backward()
                self.optimizer_snn.step()
            i+=1

        return snn_loss.item()

    def train_siamese_network_2(self, x, y=None):
        """
        Function to train Siamese Network with support/query couples.
        """
        self.subnetwork.train()
        z_support, z_query = self.parse_feature(x, is_feature=False)

        # Detach ensures we don't change the weights in main training process
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1).detach().to(self.device)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).detach().to(self.device)

        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]

        if y is None:  # Classification
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
            y_support = Variable(y_support.to(self.device))
        else:  # Regression
            y_support = y[:, :self.n_support]
            y_support = y_support.contiguous().view(self.n_way * self.n_support, -1).to(self.device)
    
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(10):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                self.optimizer_snn.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)])
                # z_batch_s = z_proto[selected_id]
                # z_batch_q = z_query[selected_id]
                z_batch_s = z_support[selected_id]
                z_batch_q = z_query[selected_id]
                #y_batch = y_query[selected_id]
                y_batch = y_support[selected_id]
                loss_snn = self.set_forward_snn_loss([z_batch_s, z_batch_q], y_batch)
                # loss.backward(retain_graph=True)
                loss_snn.backward()
                self.optimizer_snn.step()

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
            self.train_siamese_network_2(x.to(self.device))
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
        Function to compute the forward loss for the main classification task.
        """
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)


    def set_forward_snn_loss(self, x, labels, is_feature=False):
        """
        Function to compute the forward loss for training of the Siamese network.
        """

        euclidean_dist_embeddings = self.set_forward_snn(x)
        
        # parameter of the loss to experiment with
        margin = 0.5

        first_term = ((1.0-labels.float())*euclidean_dist_embeddings**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0.0),margin-euclidean_dist_embeddings)**2
        contrastive_loss = first_term + second_term
        # Use mean to get the average loss overall classes, want to minimize the loss for all distances
        #print("contrastive loss", contrastive_loss)
        return contrastive_loss.mean()

    def set_forward_snn(self, x):
        """
        Function to set forward pass through the Siamese network.
        """

        output1, output2 = self.subnetwork(x[0]), self.subnetwork(x[1])
        euclidean_dist_embeddings = euclidean_dist(output1,output2)

        return euclidean_dist_embeddings
    
    def split_data_into_pairs(self, data, labels):
        """
        Preprocess data and labels for Siamese Neural Network training.
        support for training siamese, query for validation

        """
        pairs, pairLabels = [], []
        classes = np.unique(labels, axis=0)
        pair_snn_01, pair_snn_02 = [], []

        for sampleIdx in range(len(labels)):
            sample = data[sampleIdx]
            label = labels[sampleIdx]
            # randomly pick a sample that belongs to the same class
            posSampleIdx = np.random.choice(np.where(labels == label)[0])
            posSample = data[posSampleIdx]
            # create a positive pair
            pair_snn_01.append(sample)
            pair_snn_02.append(posSample)
            pairLabels.append(1)

            negIdx = np.where(labels != label)[0]
            negSample = data[np.random.choice(negIdx)]
            # prepare a negative pair of samples and update our lists
            pair_snn_01.append(sample)
            pair_snn_02.append(negSample)
            pairLabels.append(0)

        pairs.append(torch.stack(pair_snn_01))
        pairs.append(torch.stack(pair_snn_02))

        return pairs, torch.tensor(pairLabels)

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

