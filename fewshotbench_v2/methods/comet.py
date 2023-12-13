# This code is modified from https://github.com/jakesnell/prototypical-networks 

#import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class COMET(MetaTemplate):
    def __init__(self, backbone,  n_way, n_support):
        super(COMET, self).__init__( backbone,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        #change distance here
        dists = manhattan_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

# Different distance metrics

def euclidean_dist( x, y):
    '''
    Compute the Euclidean distance between two sets of vectors x and y.
    x: Tensor of shape (N, D), where N is the number of samples and D is the dimensionality.
    y: Tensor of shape (M, D), where M is the number of samples and D is the dimensionality.
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def euclidean_dist_correct(x, y):
    '''
    Compute the Square Rooted Euclidean distance between two sets of vectors x and y.
    x: Tensor of shape (N, D), where N is the number of samples and D is the dimensionality.
    y: Tensor of shape (M, D), where M is the number of samples and D is the dimensionality.
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.sqrt(torch.pow(x - y, 2).sum(2))

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




