import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from methods.meta_template import MetaTemplate

class SiameseMAML(MetaTemplate):
    def __init__(self, input_shape, n_way, n_support, n_task, task_update_num, inner_lr, approx=False):
        super(SiameseMAML, self).__init__(None, n_way, n_support, change_way=False)

        self.n_task = n_task
        self.task_update_num = task_update_num
        self.inner_lr = inner_lr
        self.approx = approx

        if n_way == 1:
            self.type = "regression"
            self.loss_fn = nn.MSELoss()
        else:
            self.type = "classification"
            self.loss_fn = nn.CrossEntropyLoss()

        # Create Siamese subnetworks
        self.subnetwork1 = self.create_subnetwork(input_shape)
        self.subnetwork2 = self.create_subnetwork(input_shape)

        # Set the weights of the second subnetwork to be equal to those of the first
        self.subnetwork2.set_weights(self.subnetwork1.get_weights())

        # Define inputs for the Siamese network
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)

        # Process inputs through the subnetworks
        output1 = self.subnetwork1(input1)
        output2 = self.subnetwork2(input2)

        # Create a model that connects the inputs and the output
        self.siamese_model = Model([input1, input2], [output1, output2])

        # Compile model with contrastive loss and Adam optimizer
        self.siamese_model.compile(loss=lambda y_true, y_pred: -tf.reduce_sum(tf.square(y_pred[0] - y_pred[1])),
                                   optimizer=Adam())
        
        # TODO: include classifier?

    def create_subnetwork(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = Dense(512, activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(32, activation='relu')(x)
        model = Model(input_layer, output_layer)
        return model

    def set_forward(self, x, y=None):
        # Process inputs through the Siamese network
        output1 = self.subnetwork1.predict(x[0])
        output2 = self.subnetwork2.predict(x[1])

        # Compute contrastive loss
        contrastive_loss = -np.sum(np.square(output1 - output2))

        # Perform task updates
        for task_step in range(self.task_update_num):
            grads = np.gradient(contrastive_loss, self.subnetwork1.trainable_variables)
            fast_parameters = [w - self.inner_lr * g for w, g in zip(self.subnetwork1.trainable_variables, grads)]
            self.subnetwork1.set_weights(fast_parameters)

        # Process query set through the updated subnetwork
        updated_output1 = self.subnetwork1.predict(x[2])

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

        loss = self.loss_fn(scores, y_b_i)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []

        for i, (x, y) in enumerate(train_loader):
            if task_count == 0:
                optimizer.zero_grad()

            # list containing support set, query set, and corresponding labels
            x_support, x_query, y_batch = x

            # Labels are assigned later if classification task
            if self.type == "classification":
                y_batch = None

            # Convert NumPy arrays to TensorFlow tensors?
            x_support_tf = tf.convert_to_tensor(x_support, dtype=tf.float32)
            x_query_tf = tf.convert_to_tensor(x_query, dtype=tf.float32)
            y_batch_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)

            # Set up input for the Siamese network
            inputs = [x_support_tf[0:1], x_support_tf[1:2], x_query_tf[0:1]]

            # Calculate loss using set_forward_loss method
            loss = self.set_forward_loss(inputs, y_batch_tf)

            avg_loss += loss
            loss_all.append(loss)
            task_count += 1

            if task_count == self.n_task:
                avg_loss /= self.n_task
                avg_loss.backward()
                optimizer.step()
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

            # Convert NumPy arrays to TensorFlow tensors?
            x_support_tf = tf.convert_to_tensor(x_support, dtype=tf.float32)
            x_query_tf = tf.convert_to_tensor(x_query, dtype=tf.float32)
            y_batch_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)

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

