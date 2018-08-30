import mxnet as mx
from mxnet import nd, gluon


class CnnTextClassifier(gluon.nn.HybridSequential):

    def __init__(self, num_filters, fully_connected, dropout, num_outputs):
        super().__init__()
        with self.name_scope():
            self.conv1 = gluon.nn.Conv1D(channels=num_filters, kernel_size=7, activation='relu')
            self.pool1 = gluon.nn.MaxPool1D(pool_size=3, strides=3)
            self.conv2 = gluon.nn.Conv1D(channels=num_filters, kernel_size=7, activation='relu')
            self.pool2 = gluon.nn.MaxPool1D(pool_size=3, strides=3)
            self.conv3 = gluon.nn.Conv1D(channels=num_filters, kernel_size=3, activation='relu')
            self.conv4 = gluon.nn.Conv1D(channels=num_filters, kernel_size=3, activation='relu')
            self.conv5 = gluon.nn.Conv1D(channels=num_filters, kernel_size=3, activation='relu')
            self.conv6 = gluon.nn.Conv1D(channels=num_filters, kernel_size=3, activation='relu')
            self.pool3 = gluon.nn.MaxPool1D(pool_size=3, strides=3)
            self.flatten = gluon.nn.Flatten()
            self.dense = gluon.nn.Dense(fully_connected, activation='relu')
            self.drop = gluon.nn.Dropout(dropout)
            self.fc = gluon.nn.Dense(fully_connected, activation='relu')
            self.drop2 = gluon.nn.Dropout(dropout)
            self.output = gluon.nn.Dense(num_outputs)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        c = self.conv1(x)
        p = self.pool1(c)
        c = self.conv2(p)
        p = self.pool2(c)
        c = self.conv3(p)
        c = self.conv4(c)
        c = self.conv5(c)
        c = self.conv6(c)
        p = self.pool3(c)
        f = self.flatten(p)
        d = self.dense(f)
        drop = self.drop(d)
        fc = self.fc(drop)
        drop = self.drop2(fc)
        return self.output(drop)


if __name__ == "__main__":
    """
    Run unit-test
    """
    net = CnnTextClassifier(num_filters=10, fully_connected=10, dropout=0.5, num_outputs=5)

    x = nd.random.uniform(shape=(128, 69, 1014))
    net.initialize()
    y = net(x)
    assert y.shape == (128, 5)
    nd.waitall()
    print("Unit-test success!")
