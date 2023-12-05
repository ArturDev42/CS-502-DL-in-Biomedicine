import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from methods.meta_template import MetaTemplate


class SiameseMAML(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(SiameseMAML, self).__init__(backbone, n_way, n_support)
        print("SiameseMAML(MetaTemplate)")

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

    def forward(self, x):
        print("forward(self,x)")
        output1 = self.subnetwork1(x[0])
        output2 = self.subnetwork2(x[1])
        return output1, output2

    def set_forward(self, x, y=None):
        print("set_forward(self,x)")
        # Process inputs through the Siamese network
        output1, output2 = self(x)

        # AD: Squared Euclidean distance between output1 and output2?
        # AD: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss
        # Compute contrastive loss
        # TODO: here only distance - change to loss
        # (1−Y)12(Dw)2+12{max(0,margin−Dw)}2
        contrastive_loss = -torch.sum((output1 - output2) ** 2)

        # Perform task updates
        for task_step in range(self.task_update_num):
            grads = torch.autograd.grad(contrastive_loss, self.subnetwork1.parameters(), create_graph=True)
            fast_parameters = [w - self.inner_lr * g for w, g in zip(self.subnetwork1.parameters(), grads)]
            for param, fast_param in zip(self.subnetwork1.parameters(), fast_parameters):
                param.data = fast_param.data

        # Process query set through the updated subnetwork
        updated_output1, _ = self(x[2])

        return updated_output1

    def set_forward_loss(self, x, y=None):
        # Process inputs through set_forward
        updated_output = self.set_forward(x, y)

        if y is None:  # Classification task
            y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query)))
        else:  # Regression task
            y_var = Variable(y)
            y_b_i = y_var[:, self.n_support:].contiguous().view(self.n_way * self.n_query, *y.size()[2:])

        if torch.cuda.is_available():
            y_b_i = y_b_i.cuda()

        loss = self.loss_fn(updated_output, y_b_i)

        return loss

# todo: make optimizer variable
    def train_loop(self, epoch, train_loader):
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []

        for i, (x, y) in enumerate(train_loader):
            if task_count == 0:
                self.optimizer.zero_grad()

            # list containing support set, query set, and corresponding labels
            x_support, x_query, y_batch = x

            # Labels are assigned later if classification task
            if self.type == "classification":
                y_batch = None

            # Convert NumPy arrays to PyTorch tensors
            x_support = torch.from_numpy(x_support).float()
            x_query = torch.from_numpy(x_query).float()
            y_batch = torch.from_numpy(y_batch).float()

            # Set up input for the Siamese network
            inputs = [x_support[0:1], x_support[1:2], x_query[0:1]]

            # Calculate loss using set_forward_loss method
            loss = self.set_forward_loss(inputs, y_batch)

            avg_loss += loss.item()
            loss_all.append(loss.item())
            task_count += 1

            if task_count == self.n_task:
                avg_loss /= self.n_task
                loss.backward()
                self.optimizer.step()
                task_count = 0
                loss_all = []

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def test_loop(self, test_loader, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            # list containing support set, query set, and corresponding labels
            x_support, x_query, y_batch = x

            if self.type == "classification":
                # should be there from meta template
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
            else:
                # Use pearson correlation
                acc_all.append(self.correlation(x, y))

            # Convert NumPy arrays to PyTorch tensors
            x_support = torch.from_numpy(x_support).float()
            x_query = torch.from_numpy(x_query).float()
            y_batch = torch.from_numpy(y_batch).float()

            acc_all = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)

        if self.type == "classification":
            print('%d Accuracy = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        else:
            # print correlation
            print('%d Correlation = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
