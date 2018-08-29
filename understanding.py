import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import time


batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000

#########################################
# Read in and transform the mnist dataset
#########################################


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

# for i, (data, label) in enumerate(train_data):
#     if i < 1:
#         print("Epoch {} data {} label {}".format(i, data, label))

####################
# Define the network
####################


class Model(gluon.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)  # this calls init on the parent class (gluon.Block)


        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(64)
            self.dense1 = gluon.nn.Dense(64)
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        """
        takes some NDArray input x and generates an NDArray output
        Because the output and input are related to each other via NDArray operations,
        MXNet can take derivatives through the block automatically
        """
        # print("Input data: {}".format(x))
        x = self.dense0(x)
        # print("Hidden layer output: {}".format(x))
        x = nd.relu(x)
        # print("Relu act output: {}".format(x))
        x = self.dense1(x)
        # print("Hidden layer output: {}".format(x))
        x = nd.relu(x)
        # print("Hidden layer output: {}".format(x))
        x = self.dense2(x)
        # print("Hidden layer output: {}".format(x))
        return x


# instantiate model class
net = Model()

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
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# create a training loop
cumulative_time = 0
for e in range(5):
    start_time = time.time()
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.reshape((-1, 784))
        with autograd.record():  # keep track of predictions and loss
            output = net(data)
            loss = sm_loss(output, label)

        loss.backward()  # back-propogate loss through network

        optimizer.step(batch_size=data.shape[0])  # update network parameters (updates depend on the batch size)

        # sum the loss for each training example in the batch & convert from ndarray shape (1,) to float
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net, mx.cpu())
    train_accuracy = evaluate_accuracy(train_data, net, mx.cpu())
    l = cumulative_loss / num_examples  # average cumulative loss per training example
    epoch_time = time.time() - start_time
    cumulative_time += epoch_time

    print("Epoch {} Time: {} Loss: {}, Train_acc {}, Test_acc {}".format(e, epoch_time, l, train_accuracy, test_accuracy))
print("Imperitive Train Time = 75.63905787467957, Symbolic Train Time = {}".format(cumulative_time))
