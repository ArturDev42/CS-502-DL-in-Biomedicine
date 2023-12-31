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

    def train_siamese_network(self, x):
        '''
        Function to train our Siamese Network.

        Args:
            x: Input for our Siamese Neural Network

        Returns:
            Tensor: Output contrastive loss

        '''
        self.subnetwork.train()
        freq = 10

        x, _ = self.parse_feature(x, is_feature=False)

        pairs, pair_labels = self.split_data_into_pairs(x)

        # Convert pairs and labels to PyTorch variables
        pairs = [Variable(pair.to(self.device)) for pair in pairs]
        pair_labels = Variable(pair_labels.to(self.device))

        self.optimizer_snn.zero_grad()

        # Forward pass for Siamese network training
        i = 1
        for pair0, pair1, label in zip(pairs[0], pairs[1], pair_labels):
            pair0 = pair0.reshape(1, -1)
            pair1 = pair1.reshape(1, -1)
            snn_loss = self.set_forward_snn_loss([pair0, pair1], label)
            if i % freq == 0:
                snn_loss.backward()
                self.optimizer_snn.step()
            i+=1

        return snn_loss.item()

    # train loop
    def train_loop(self, epoch, train_loader, optimizer):
        '''
        Function to train our whole Model - Difference to Meta Template: added Siamese Network training

        Args:
            epoch (int): Current epoch number.
            train_loader (DataLoader): DataLoader for training data.
            optimizer (Optimizer): Optimizer for updating model parameters.
            
        '''
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            # train the siamese network
            self.train_siamese_network(x.to(self.device))
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
        '''
        Computes the loss for training our entire Model.

        Args:
            x ([Tensor, Tensor]): Input for our Siamese Neural Network

        Returns:
            Tensor: Output loss

        '''
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)


    def set_forward_snn_loss(self, x, labels):
        '''
        Computes the contrastive loss for training the Siamese network.

        Args:
            x ([Tensor, Tensor]): Input for our Siamese Neural Network
            labels (Tensor): Labels indicating whether the input tuples belong to the same class (1) or different classes (0)

        Returns:
            Tensor: Output contrastive loss
        '''

        euclidean_dist_embeddings = self.set_forward_snn(x)
        
        # parameter of the loss to experiment with
        margin = 1

        first_term = ((1.0-labels.float())*euclidean_dist_embeddings**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0.0),margin-euclidean_dist_embeddings)**2
        contrastive_loss = first_term + second_term
        
        return contrastive_loss.mean()

    def set_forward_snn(self, x):
        '''
        Function to set forward pass through the Siamese network.

        Args:
            x ([Tensor, Tensor]): Input pairs for our Siamese Neural Network, where x[0] is the first element, and x[1] is the second element.

        Returns:
            Tensor: Euclidean distances between the output embeddings of the Siamese network.
        '''

        output1, output2 = self.subnetwork(x[0]), self.subnetwork(x[1])
        # you can change the distance here
        euclidean_dist_embeddings = euclidean_dist(output1,output2)

        return euclidean_dist_embeddings

    def split_data_into_pairs(self, data):
        '''
        Preprocess data and labels for Siamese Neural Network training.

        Args:
            data (Tensor): Input support data of shape [num_way, num_support, n_dim]
        
        Returns:
            Tuple[[Tensor, Tensor], Tensor]: A tuple containing a list of pairs of samples and corresponding pair labels.

        '''
        pairs, pairLabels = [], []
        pair_snn_01, pair_snn_02 = [], []

        for c in range(data.size(0)):
            support = data[c]
            for sampleIdx in range(support.size(0)):
                sample = support[sampleIdx]
                # randomly pick a sample that belongs to the same class
                posSampleIdx = np.random.choice(support.size(0))
                posSample = support[posSampleIdx]
                # create a positive pair
                pair_snn_01.append(sample)
                pair_snn_02.append(posSample)
                pairLabels.append(1)

                negIdx = np.where(np.arange(data.size(0)) != c)[0]
                # In case of 1-way learning - no negative examples possible
                if len(negIdx) != 0:
                    # pick a random sample from a different class
                    negClass = data[np.random.choice(negIdx)]
                    negSampleIdx = np.random.choice(negClass.size(0))
                    negSample = negClass[negSampleIdx]
                    # prepare a negative pair of samples and update our lists
                    pair_snn_01.append(sample)
                    pair_snn_02.append(negSample)
                    pairLabels.append(0)

        pairs.append(torch.stack(pair_snn_01))
        pairs.append(torch.stack(pair_snn_02))

        return pairs, torch.tensor(pairLabels)


def euclidean_dist(x, y):
    '''
    Compute the Euclidean distance between two sets of vectors x and y.
    x: Tensor of shape (N, D), where N is the number of samples and D is the dimensionality.
    y: Tensor of shape (M, D), where M is the number of samples and D is the dimensionality.
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2).sqrt()


def manhattan_dist(x, y):
    '''
    Compute the Manhattan (L1) distance between two sets of vectors x and y.
    x: Tensor of shape (N, D), where N is the number of samples and D is the dimensionality.
    y: Tensor of shape (M, D), where M is the number of samples and D is the dimensionality.
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.abs(x - y).sum(2)





