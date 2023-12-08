import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from methods.meta_template import MetaTemplate


class MapCell(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MapCell, self).__init__(backbone, n_way, n_support)

        self.classifier = nn.Linear(self.feat_dim, n_way)

        if n_way == 1:
            self.type = "regression"
            self.loss_fn = nn.MSELoss()
        else:
            self.type = "classification"
            self.loss_fn = nn.CrossEntropyLoss()

        # Create Siamese subnetworks
        self.subnetwork1 = self.create_subnetwork(self.feat_dim)
        self.subnetwork2 = self.create_subnetwork(self.feat_dim)

        # Set the weights of the second subnetwork to be equal to those of the first
        self.subnetwork2.load_state_dict(self.subnetwork1.state_dict())

        # Define inputs for the Siamese network
        self.siamese_model = nn.Sequential(self.subnetwork1, self.subnetwork2)

        # Compile model with contrastive loss and Adam optimizer
        self.optimizer = optim.Adam(self.siamese_model.parameters())

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

    def forward_snn(self, x):
        output1 = self.subnetwork1(x[0])
        output2 = self.subnetwork2(x[1])

        # TODO: here only distance - change to loss
        # (1−Y)12(Dw)2+12{max(0,margin−Dw)}2
        euclidean_dist_embeddings = -torch.sum((output1 - output2) ** 2)

        # snn_distance = 

        return snn_distance

    # TODO: Prepare data for SNN
    # Construct sample paris from the prototype vector set
    # Pair ech sample in the support set with all the prototype vectors
    # If two items in a pair are in the same class, label = 1, otherwise label = 0
    def construct_sample_pairs(self, z_proto, z_query):
        return None

    def set_forward(self, x, is_feature=False):
        # Slide 16, Mete-Learning Lec: Explanation of z_support, z_query 
        z_support, z_query = self.parse_feature(x, is_feature)
 
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # TODO: Logic for returning SNN distances
        # Call construct_sample_pairs()
        # Pass prepared data to SNN
        # ...  
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores
    

    def set_forward_loss(self, x):
        # TODO: Same type of change as for set_forward() necessary?

        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        print("self.loss_fn(scores, y_query )")
        print("scores", scores)
        print("y_query", y_query)

        return self.loss_fn(scores, y_query )


    # def set_forward(self, x, y=None):
    #     # Process inputs through the Siamese network
    #     output1, output2 = self(x)

    #     # AD: Squared Euclidean distance between output1 and output2?
    #     # AD: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss
    #     # Compute contrastive loss
    #     # TODO: here only distance - change to loss
    #     # (1−Y)12(Dw)2+12{max(0,margin−Dw)}2
    #     contrastive_loss = -torch.sum((output1 - output2) ** 2)

    #     # Perform task updates
    #     for task_step in range(self.task_update_num):
    #         grads = torch.autograd.grad(contrastive_loss, self.subnetwork1.parameters(), create_graph=True)
    #         fast_parameters = [w - self.inner_lr * g for w, g in zip(self.subnetwork1.parameters(), grads)]
    #         for param, fast_param in zip(self.subnetwork1.parameters(), fast_parameters):
    #             param.data = fast_param.data

    #     # Process query set through the updated subnetwork
    #     updated_output1, _ = self(x[2])

    #     return updated_output1

    # def set_forward_loss(self, x, y=None):
    #     # Process inputs through set_forward
    #     updated_output = self.set_forward(x, y)

    #     if y is None:  # Classification task
    #         y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query)))
    #     else:  # Regression task
    #         y_var = Variable(y)
    #         y_b_i = y_var[:, self.n_support:].contiguous().view(self.n_way * self.n_query, *y.size()[2:])

    #     if torch.cuda.is_available():
    #         y_b_i = y_b_i.cuda()

    #     loss = self.loss_fn(updated_output, y_b_i)

    #     return loss

