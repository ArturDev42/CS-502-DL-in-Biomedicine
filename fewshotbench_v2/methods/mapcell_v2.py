import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import torch.optim as optim

from backbones.blocks import distLinear
from methods.meta_template import MetaTemplate


class MapCell_new(MetaTemplate):

    def __init__(self, backbone, n_way, n_support, n_classes=1, loss='softmax', type='classification'):
        super(MapCell_new, self).__init__(backbone, n_way, n_support, change_way=True)
        self.feature = backbone
        self.type = type
        self.n_classes = n_classes

        if loss == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, n_classes)
            self.classifier.bias.data.fill_(0)
        elif loss == 'dist':  # Baseline ++
            self.classifier = distLinear(self.feature.final_feat_dim, n_classes)

        self.loss_type = loss  # 'softmax' #'dist'

        if self.type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.type == 'regression':
            self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print("model - self:", self.feature)
        self.loss_fn = nn.CrossEntropyLoss()

        # Create Siamese subnetworks

        self.subnetwork1 = self.create_subnetwork(self.feat_dim)
        self.subnetwork2 = self.create_subnetwork(self.feat_dim)

        # Set the weights of the second subnetwork to be equal to those of the first
        self.subnetwork2.load_state_dict(self.subnetwork1.state_dict())

        # Define inputs for the Siamese network
        self.siamese_model = nn.Sequential(self.subnetwork1, self.subnetwork2)

        # Compile model with contrastive loss and Adam optimizer
        self.optimizer_snn = optim.RMSprop(
            self.siamese_model.parameters(), 
            lr=0.01, 
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
            nn.Linear(512, input_shape),
            nn.ReLU()
        )

        return model

    def forward(self, x):
        if isinstance(x, list):
            x = [Variable(obj.cuda()) for obj in x]
        else:
            x = Variable(x.cuda())

        out = self.feature.forward(x)
        out = self.subnetwork1(out)
        # dists = self.set_forward_snn(out)
        # scores = -dists
        if self.classifier != None:
            scores = self.classifier.forward(out)
        return scores

    def set_forward_loss(self, x, y):
        scores = self.forward(x)
        #print(scores.shape)
        if self.type == 'classification':
            y = y.long().cuda()
        else:
            y = y.cuda()

        return self.loss_fn(scores, y)

    def set_forward_snn_loss(self, x, labels, is_feature=False):
        #print("set_forward_snn_loss(self, x, labels, is_feature=False)")
        x[0] = self.feature.forward(x[0].to(self.device))
        x[1] = self.feature.forward(x[1].to(self.device))

        euclidean_dist_embeddings = self.set_forward_snn(x)
        # print("labels", labels)
        # print("euclidean_dist before", euclidean_dist_embeddings)

        euclidean_dist_embeddings = euclidean_dist_embeddings.mean(1)

        #print("euclidean_dist after", euclidean_dist_embeddings)
        # parameter of the loss to experiment with
        margin = 1

        labels = torch.tensor(labels)
        labels = labels.to(self.device)
        one_vector = torch.ones_like(labels)
        one_vector = one_vector.to(self.device)

        # print(labels)
        # print(one_vector)
        # print(euclidean_dist_embeddings)

        margin_vector = torch.full(euclidean_dist_embeddings.size(), margin).to(self.device)
        first_term = torch.matmul(one_vector.float()-labels.float(),euclidean_dist_embeddings**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0),margin_vector-euclidean_dist_embeddings)**2
        contrastive_loss = first_term + second_term
        #print("contrastive_loss", contrastive_loss.size())
        return contrastive_loss.mean()

    def set_forward_snn(self, x):
        #print("set_forward_snn(self, x)")

        """
        x[0]:
        [tensor([-0.4559,  1.7199, -0.3444, -0.1739, -0.2226], device='cuda:0'),
         tensor([ 0.9357,  1.1822, -0.3444, -0.1739, -0.2226], device='cuda:0')]
        
        x[1]:
        [tensor([-0.4559,  1.7199, -0.3444, -0.1739, -0.2226], device='cuda:0'), 
        tensor([ 0.5071, -0.4544, -0.3293, -0.1739, -0.2226], device='cuda:0')]
        """

        output1 = self.subnetwork1(x[0])
        output2 = self.subnetwork2(x[1])

        # print("output1", output1.size())
        # print("output1", output2.size())
        euclidean_dist_embeddings = torch.cdist(output1,output2,p=2)
        #euclidean_dist_embeddings = (output1-output2).pow(2).sum(1).sqrt()

        #print("euclidean_dist_embeddings", euclidean_dist_embeddings.size())
        return euclidean_dist_embeddings

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):

            data, labels = self.preprocess_siamese_data(x, y)
            snn_train, snn_test = data
            snn_train_labels, snn_test_labels = labels

            self.optimizer_snn.zero_grad()
            loss_snn = self.set_forward_snn_loss(snn_train, snn_train_labels)
            loss_snn.backward(retain_graph=True)
            self.optimizer_snn.step()

            optimizer.zero_grad()

            loss = self.set_forward_loss(snn_test, snn_test_labels)

            # if self.change_way:
            #     self.n_way = self.n_classes
            
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss/train": avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=None):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
            else:
                self.n_query = x.size(1) - self.n_support

            if self.type == "classification":
                correct_this, count_this = self.correct(x)
                print("correct:", correct_this)
                acc_all.append(correct_this / count_this * 100)
            else:
                # Use pearson correlation
                acc_all.append(self.correlation(x, y))

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

    def set_forward(self, x, y=None):
        z_support, z_query = self.parse_feature(x, is_feature=False)

        # Detach ensures we don't change the weights in main training process
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1).detach().to(self.device) 
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).detach().to(self.device)

        if y is None:  # Classification
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
            y_support = Variable(y_support.to(self.device))
        else:  # Regression
            y_support = y[:, :self.n_support]
            y_support = y_support.contiguous().view(self.n_way * self.n_support, -1).to(self.device)
            # y_support = y_support.contiguous().view(self.n_way * y.size(1), -1)

        # if self.loss_type == 'softmax':
        #     linear_clf = nn.Linear(self.feat_dim, self.n_way)
        # elif self.loss_type == 'dist':
        #     linear_clf = distLinear(self.feat_dim, self.n_way)
        # else:
        #     raise ValueError('Loss type not supported')

        # linear_clf = linear_clf.to(self.device)

        # set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
        #                                 weight_decay=0.001)

        # loss_function = self.loss_fn.to(self.device)

        # batch_size = 4
        # support_size = self.n_way * self.n_support
        # for epoch in range(100):
        #     rand_id = np.random.permutation(support_size)
        #     for i in range(0, support_size, batch_size):
        #         set_optimizer.zero_grad()
        #         selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
        #         z_batch = z_support[selected_id]
        #         y_batch = y_support[selected_id]
        #         # TODO: y through network
        #         with torch.no_grad():
        #             scores = self.subnetwork1(z_batch)
        #         scores = linear_clf(scores)
        #         loss = loss_function(scores, y_batch)
        #         # loss.backward(retain_graph=True)
        #         loss.backward()
        #         set_optimizer.step()

        dists = self.set_forward_snn([z_query, z_proto])
        print("dists:", dists)
        # scores = linear_clf(-dists)
        scores = -dists
        print("scores:", scores)
        return scores

    def preprocess_siamese_data(self, data, labels):
            """
            Preprocess data and labels for Siamese Neural Network training.
            support for training siamese, query for validation
            """

            query, query_labels = data.to(self.device), labels.to(self.device)

            # TODO: support query split based on features? Was genau haben wir hier, wieso preprocessing notwendig? Von meta_template...
            if isinstance(data, list):
                data = [Variable(obj.to(self.device)) for obj in data]
            else: 
                data = Variable(data.to(self.device))

            #print("data before ", data.size())

            z_all = data
            # TODO; smarter split: wie n_support umgehen?

            # print("data ", data.size())
            # print("z_all ", z_all.size())

            z_support = z_all[:self.n_support]
            z_query = z_all[self.n_support:]

            z_s_labels = labels[:self.n_support]
            z_q_labels = labels[self.n_support:]

            # TODO: query sollte nur query sein und labels
            # vergleichen mit protos - protos müssen erstellt werden
            # paare nicht so wie support erstellen

            support, support_labels = self.split_data_into_pairs(z_support, z_s_labels)

            pairs, pairLabels = [support, query], [support_labels, query_labels]

            return pairs, pairLabels

    def split_data_into_pairs(self, data, labels):
        """
        Preprocess data and labels for Siamese Neural Network training.
        support for training siamese, query for validation

        """
        #print("split_data_into_pairs(self, data, labels")
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
            if(negIdx.size != 0):
                negSample = data[np.random.choice(negIdx)]
                # prepare a negative pair of samples and update our lists
                pair_snn_01.append(sample)
                pair_snn_02.append(negSample)
                pairLabels.append(0)

            #print("sample shape", sample.size())

        pairs.append(torch.stack(pair_snn_01))
        pairs.append(torch.stack(pair_snn_02))

        return pairs, torch.tensor(pairLabels)
