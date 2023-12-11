import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
import wandb

class MapCell(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MapCell, self).__init__(backbone, n_way, n_support)

        print("model - self:", self.feature)

        self.classifier = nn.Linear(self.feat_dim, n_way)
        self.loss_fn = nn.CrossEntropyLoss()

        # Create Siamese subnetworks

        # print("self.feat_dim",self.feat_dim)

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
            nn.Linear(512, 32),
            nn.ReLU()
        )
        return model

    def correct(self, x):
        scores = self.subnetwork1(x)
        print("scores: ", scores)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward_snn_loss(self, x, labels, is_feature=False):
        print("set_forward_snn_loss(self, x, labels, is_feature=False)")

        x[0] = x[0].mean(1)  # the shape of z is [n_data, n_dim]
        x[1] = x[1].mean(1)  # the shape of z is [n_data, n_dim]
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        euclidean_dist_embeddings = self.set_forward_snn(x)
        print("labels", labels)

        # parameter of the loss to experiment with
        margin = 1
        print(self.device)

        labels = torch.tensor(labels)
        labels = labels.to(self.device)
        one_vector = torch.ones_like(labels)
        one_vector = one_vector.to(self.device)

        print(labels)
        print(one_vector)
        print(euclidean_dist_embeddings)

        margin_vector = torch.full(euclidean_dist_embeddings.size(), margin).to(self.device)
        print(margin_vector.shape)

        print(margin_vector.dtype)
        print(labels.dtype)
        print(one_vector.dtype)
        print(euclidean_dist_embeddings.dtype)
        first_term = torch.matmul(one_vector.float()-labels.float(),euclidean_dist_embeddings**2)*0.5

        second_term = 0.5*torch.maximum(torch.tensor(0),margin_vector-euclidean_dist_embeddings)**2
        contrastive_loss = first_term + second_term
        print("contrastive_loss", contrastive_loss.size())
        return contrastive_loss.mean()

    def set_forward_snn(self, x):
        print("set_forward_snn(self, x)")
        #print("x[0]", x[0])
        #print("x[1]", x[1])

        """
        x[0]:
        [tensor([-0.4559,  1.7199, -0.3444, -0.1739, -0.2226], device='cuda:0'),
         tensor([ 0.9357,  1.1822, -0.3444, -0.1739, -0.2226], device='cuda:0')]
        
        x[1]:
        [tensor([-0.4559,  1.7199, -0.3444, -0.1739, -0.2226], device='cuda:0'), 
        tensor([ 0.5071, -0.4544, -0.3293, -0.1739, -0.2226], device='cuda:0')]
        """

        output1 = self.subnetwork1(x[0].to(self.device))
        output2 = self.subnetwork2(x[1].to(self.device))

        print("output1", output1.size())
        print("output1", output2.size())
        #euclidean_dist_embeddings = torch.cdist(output1,output2,p=2)
        euclidean_dist_embeddings = (output1-output2).pow(2).sum(1).sqrt()

        print("euclidean_dist_embeddings", euclidean_dist_embeddings.size())
        return euclidean_dist_embeddings

    def set_forward(self, query_pair, is_feature=False):
        print("set_forward(self, query_pair, is_feature=False)")
        print("query_pair")
        print(query_pair[0].size())
    
        # query_pair[0] = query_pair[0].contiguous().view(self.n_way * self.n_query, -1)
        # query_pair[1] = query_pair[1].contiguous().view(self.n_way * self.n_query, -1)

        print("query_pair after")
        print(query_pair[0].size())

        # TODO: eigene forward function die nur ein subnetzwerk benutzt?
        dists = self.set_forward_snn(query_pair)
        scores = -dists
        return scores
    
    def set_forward_loss(self, x, y):
        print("set_forward_loss(self, x)")
        # y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        # y_query = Variable(y_query.cuda())

        print("x", x[0].size())
        scores = self.set_forward(x)

        # print("self.loss_fn(scores, y_query )")
        print("scores", scores)
        print(scores.size())
        # print("y_query", y_query)

        return self.loss_fn(scores, y.to(self.device))


    def train_loop(self, epoch, train_loader, optimizer):
        print("train_loop(self, epoch, train_loader, optimizer)")
        print_freq = 10

        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                print("x", x)
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)

            # idea - split the data into support and query to "pretrain" siamese

            # print("x")
            # print(x)
            # print(x.size())

            # print("y")
            # print(y)
            # print(y.size())

            data, labels = self.preprocess_siamese_data(x, y)
            support, query = data
            support_labels, query_labels = labels
            
            self.optimizer_snn.zero_grad()
            print("support", support[0].size())
            loss_snn = self.set_forward_snn_loss(support, support_labels)
            loss_snn.backward(retain_graph=True)
            self.optimizer_snn.step()

            optimizer.zero_grad()
            print("query before set_forward_loss()", query[0].size())

            loss = self.set_forward_loss(query, query_labels)
            loss.backward()
            optimizer.step()
            
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})

    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
        
            # Split data into pairs
            #data = self.split_data_into_pairs_test(x)
            data = x
            if isinstance(data, list):
                data = [Variable(obj.to(self.device)) for obj in data]
            else: 
                data = Variable(data.to(self.device))

            print("data before ", data.size())

            if isinstance(data, list):
                data = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in data]
            else: data = data.contiguous().view(self.n_way * (self.n_support + self.n_query), *data.size()[2:])
            data = self.feature.forward(data)

            correct_this, count_this = self.correct(data)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def preprocess_siamese_data(self, data, labels):
        """
        Preprocess data and labels for Siamese Neural Network training.
        support for training siamese, query for validation
        """

        # TODO: support query split based on features? Was genau haben wir hier, wieso preprocessing notwendig? Von meta_template...
        if isinstance(data, list):
            data = [Variable(obj.to(self.device)) for obj in data]
        else: 
            data = Variable(data.to(self.device))

        print("data before ", data.size())

        if isinstance(data, list):
            data = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in data]
        else: data = data.contiguous().view(self.n_way * (self.n_support + self.n_query), *data.size()[2:])
        z_all = self.feature.forward(data)

        print("z_all before ", z_all.size())
        z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        #z_all = data

        print("data ", data.size())
        print("z_all ", z_all.size())

        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        z_s_labels = labels[:, :self.n_support]
        z_q_labels = labels[:, self.n_support:]

        # TODO: query sollte nur query sein und labels
        # vergleichen mit protos - protos müssen erstellt werden
        # paare nicht so wie support erstellen

        support = []
        query = []

        for i in range(2):
            if i==0:
                data = z_support
                labels = z_s_labels
            else :
                data = z_query
                labels = z_q_labels
            
            # print("INPUT to self.split_data_into_pairs(data, labels):")
            # print(data)
            # print(data.size())
            # print(labels.size())

            pairs, pairLabels = self.split_data_into_pairs(data, labels)

            if i==0:
                support = pairs
                support_labels = pairLabels
            else :
                query = pairs
                query_labels = pairLabels

        pairs, pairLabels = [support, query], [support_labels, query_labels]

        return pairs, pairLabels

    def split_data_into_pairs(self, data, labels):
        """
        Preprocess data and labels for Siamese Neural Network training.
        support for training siamese, query for validation

        """
        print("split_data_into_pairs(self, data, labels")
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

            print("sample shape", sample.size())

        pairs.append(torch.stack(pair_snn_01))
        pairs.append(torch.stack(pair_snn_02))

        return pairs, torch.tensor(pairLabels)

    def split_data_into_pairs_test(self, data):

        if isinstance(data, list):
            data = [Variable(obj.to(self.device)) for obj in data]
        else: 
            data = Variable(data.to(self.device))

        print("data before ", data.size())

        if isinstance(data, list):
            data = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in data]
        else: data = data.contiguous().view(self.n_way * (self.n_support + self.n_query), *data.size()[2:])
        data = self.feature.forward(data)

        pairs = []
        pair_snn_01, pair_snn_02 = [], []

        for sampleIdx in range(len(data)):
            sample = data[sampleIdx]
            # randomly pick a sample that belongs to the same class
            pairSampleIdx = np.random.choice(len(data)-1)
            pairSample = data[pairSampleIdx]
            pair_snn_01.append(sample)
            pair_snn_02.append(pairSample)
        pairs.append(torch.stack(pair_snn_01))
        pairs.append(torch.stack(pair_snn_02))

        return pairs

# TODO: correct anpassen 
