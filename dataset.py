# read in pickle file, preprocess to a format that a dataloader can read extremely efficiently
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import pandas as pd
import multiprocessing
import time

train_df = pd.read_pickle('./data/ag_news/train.pickle')
test_df = pd.read_pickle('./data/ag_news/test.pickle')


#####################################
# Define the dataset and data loaders
#####################################


class UtteranceDataset(gluon.data.ArrayDataset):
    """
    preprocesses text
    """
    def __init__(self, data, labels, alphabet, feature_len, alphabet_index):
        super().__init__(data, labels)  # initialize the parent class with data and labels
        self.alphabet = alphabet
        self.feature_len = feature_len
        self.alphabet_index = alphabet_index

    def encode(self, text):
        encoded = np.zeros([len(self.alphabet), self.feature_len], dtype='float32')
        review = text.lower()[:self.feature_len - 1:-1]
        i = 0
        for letter in text:
            if i >= self.feature_len:
                break;
            if letter in self.alphabet_index:
                encoded[self.alphabet_index[letter]][i] = 1
            i += 1
        return encoded

    def __getitem__(self, idx):  # overwrites parent class method to preprocess the data as it is loaded
        return self.encode(self._data[0][idx]), self._data[1][idx]


alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
alph_idx = {letter: index for index, letter in enumerate(alph)}

train_dataset = UtteranceDataset(data=train_df.utterance.values,
                                 labels=train_df.intent.values,
                                 alphabet=alph,
                                 alphabet_index=alph_idx,
                                 feature_len=10)


test_dataset = UtteranceDataset(data=test_df.utterance.values,
                                 labels=test_df.intent.values,
                                 alphabet=alph,
                                 alphabet_index=alph_idx,
                                 feature_len=10)

train_iter = gluon.data.DataLoader(dataset=train_dataset,
                                   batch_size=128,
                                   shuffle=False,
                                   last_batch='discard',
                                   num_workers=multiprocessing.cpu_count())

test_iter = gluon.data.DataLoader(dataset=test_dataset,
                                  batch_size=128,
                                  shuffle=False,
                                  last_batch='discard',
                                  num_workers=multiprocessing.cpu_count())


# for i, (data, label) in enumerate(train_iter):
#     if i < 1:
#         print("Batch {} Data {} Label {}".format(i, data, label))

##########################
# Define the network class
##########################


class CnnTextModel(gluon.nn.HybridSequential):

    def __init__(self, NUM_FILTERS, FULLY_CONNECTED, DROPOUT_RATE, NUM_OUTPUTS):
        super().__init__()

        with self.name_scope():
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
            self.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
            self.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
            self.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
            self.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
            self.add(gluon.nn.Flatten())
            self.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
            self.add(gluon.nn.Dropout(DROPOUT_RATE))
            self.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
            self.add(gluon.nn.Dropout(DROPOUT_RATE))
            self.add(gluon.nn.Dense(NUM_OUTPUTS))


net = CnnTextModel(NUM_FILTERS=2, FULLY_CONNECTED=50, DROPOUT_RATE=0.2, NUM_OUTPUTS=len(train_df.intent.unique()))

# convert from imperitive to symbolic to increase training speed (harder to debug now)
net.hybridize()

# collect network parameters and initialize them on the cpu
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=mx.cpu(0))

# define the loss function
sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()

# create a trainer for updating the network parameters
optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': 0.1})


# evaluation metric (uses mxnet metric api)
def evaluate_accuracy(data_iterator, net, model_ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# create a training loop
cumulative_time = 0
for e in range(5):
    start_time = time.time()
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_iter):
        with autograd.record():  # keep track of predictions and loss
            output = net(data)
            loss = sm_loss(output, label)

        loss.backward()  # back-propogate loss through network

        optimizer.step(batch_size=data.shape[0])  # update network parameters (updates depend on the batch size)

        # sum the loss for each training example in the batch & convert from ndarray shape (1,) to float
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_iter, net, mx.cpu())
    train_accuracy = evaluate_accuracy(train_iter, net, mx.cpu())
    l = cumulative_loss / train_df.shape[0]  # average cumulative loss per training example
    epoch_time = time.time() - start_time
    cumulative_time += epoch_time

    print("Epoch {} Time: {} Loss: {}, Train_acc {}, Test_acc {}".format(e, epoch_time, l, train_accuracy, test_accuracy))
print("Imperitive Train Time = 75.63905787467957, Symbolic Train Time = {}".format(cumulative_time))