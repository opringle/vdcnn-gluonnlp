import argparse
import logging
import mxnet as mx
from mxnet import nd, gluon, autograd
import gluonnlp as nlp
import multiprocessing
import os

from dataset import UtteranceDataset
from model import CnnTextClassifier

# pip install in code to ensure pandas installed in docker image :(
from pip._internal import main as pipmain
pipmain(['install', 'pandas'])
import pandas as pd


def build_dataloaders(train_df, val_df, alphabet, batch_size, num_buckets, num_workers):
    """
    :param train_df: pandas dataframe of training data
    :param val_df: pandas dataframe of validation data
    :param alphabet: list of characters to have a corresponding embedding in the lookup table
    :param batch_size: number of training examples per network parameter update
    :param num_buckets: number of buckets to create for variable sequence lengths
    :param num_workers: number of cpu threads to preprocess data on
    :return: train & val data loaders for network
    """
    logging.info("Building bucketing data loaders")
    train_dataset = UtteranceDataset(data=train_df.utterance.values, labels=train_df.intent.values, alphabet=alphabet)
    test_dataset = UtteranceDataset(data=val_df.utterance.values, labels=val_df.intent.values, alphabet=alphabet)

    # Define buckets to minimize computation on padded data
    train_data_lengths = [len(x) for x in train_df.utterance.values]
    val_data_lengths = [len(x) for x in val_df.utterance.values]
    train_batch_sampler = nlp.data.sampler.FixedBucketSampler(train_data_lengths,
                                                              batch_size=batch_size,
                                                              num_buckets=num_buckets,
                                                              ratio=0.5,  # smaller sequence lengths have larger batch sizes
                                                              shuffle=True)
    val_batch_sampler = nlp.data.sampler.FixedBucketSampler(val_data_lengths,
                                                            batch_size=batch_size,
                                                            num_buckets=num_buckets,
                                                            ratio=0.5,
                                                            shuffle=False)
    logging.info("Bucket statistics: {}".format(train_batch_sampler.stats()))
    train_batches = train_batch_sampler.__len__()

    # apply padding to features & stack to labels
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack())

    train_iter = gluon.data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, batchify_fn=batchify_fn,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(dataset=test_dataset, batch_sampler=val_batch_sampler, batchify_fn=batchify_fn,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_batches


def evaluate_accuracy(data_iterator, net, ctx):
    """
    :param data_iterator: gluon data loader
    :param net: gluon hybrid sequential block
    :return: network accuracy on data
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


class TriangularSchedule:
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min(min_lr, max_lr)
        self.max_lr = max(min_lr, max_lr)
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """
    :param hyperparameters: dict of network hyperparams
    :param channel_input_dirs: dict of paths to train and val data
    :param num_gpus: number of gpus to distribute training on
    :return: gluon neural network
    """
    logging.info("Reading in data")
    train_df = pd.read_pickle(os.path.join(channel_input_dirs['train'], 'train.pickle'))
    logging.info("Loaded {} train records".format(train_df.shape[0]))
    val_df = pd.read_pickle(os.path.join(channel_input_dirs['val'], 'test.pickle'))
    logging.info("Loaded {} validation records".format(val_df.shape[0]))

    alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    logging.info("Training context: {}".format(ctx))
    batch_size = hyperparameters.get('batch_size', 128)

    logging.info("Building data loaders")
    train_iter, val_iter, updates_per_epoch = build_dataloaders(train_df=train_df,
                                                                val_df=val_df,
                                                                alphabet=alph,
                                                                batch_size=batch_size,
                                                                num_buckets=hyperparameters.get('num_buckets', 30),
                                                                num_workers=multiprocessing.cpu_count())

    logging.info("Defining network architecture")
    net = CnnTextClassifier(vocab_size=len(alph),
                            embed_size=hyperparameters.get('embed_size', 16),
                            dropout=hyperparameters.get('dropout', 0.2),
                            num_label=len(train_df.intent.unique()),
                            temp_conv_filters=hyperparameters.get('temp_conv_filters', 32),
                            blocks=hyperparameters.get('blocks', [1, 1, 1, 1]))
    logging.info("Network architecture: {}".format(net))

    if not hyperparameters.get('no_hybridize', False):
        logging.info("Hybridizing network to convert from imperitive to symbolic for increased training speed")
        net.hybridize()

    logging.info("Initializing network parameters")
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    logging.info("Defining triangular learning rate schedule")
    schedule = TriangularSchedule(min_lr=hyperparameters.get('min_lr', 0.005),
                                  max_lr=hyperparameters.get('max_lr', 0.1),
                                  cycle_length=hyperparameters.get('lr_cycle_epochs', 10) * updates_per_epoch,
                                  inc_fraction=hyperparameters.get('lr_increase_fraction', 0.4))

    optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                              optimizer_params={'momentum': hyperparameters.get('momentum', 0.9),
                                                'lr_scheduler': schedule,
                                                'wd': hyperparameters.get('l2', 0.002)})
    sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    logging.info("Training for {} epochs".format(hyperparameters.get('epochs', 10)))
    accuracies = []
    for e in range(hyperparameters.get('epochs', 10)):
        logging.info("Epoch {}: Starting Learning Rate = {:.4}".format(e, optimizer.learning_rate))
        epoch_loss = 0
        weight_updates = 0
        for data, label in train_iter:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                pred = net(data)
                loss = sm_loss(pred, label)
            loss.backward()
            optimizer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()
            weight_updates += 1
            if weight_updates % (updates_per_epoch // hyperparameters.get('epoch_batch_progress', 5)) == 0:
                logging.info("Epoch {}: Batches complete {}/{}".format(e, weight_updates, updates_per_epoch))
        train_accuracy = evaluate_accuracy(train_iter, net, ctx)
        val_accuracy = evaluate_accuracy(val_iter, net, ctx)
        accuracies.append(val_accuracy)
        logging.info("Epoch {}: Train Loss = {:.4} Train Accuracy = {:.4} Validation Accuracy = {:.4}".
                     format(e, epoch_loss / weight_updates, train_accuracy, val_accuracy))
        logging.info("Epoch {}: Best Validation Accuracy = {:.4}".format(e, max(accuracies)))


if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train a CNN for text classification")

    # Data
    group = parser.add_argument_group('Data arguments')
    group.add_argument('--train', type=str, required=True,
                       help='path to pandas pickle of training data')
    group.add_argument('--val', type=str, required=True,
                       help='path to pandas pickle of validation data')

    # Computation
    group = parser.add_argument_group('Computation arguments')
    parser.add_argument('--gpus', type=int, default=0,
                        help='num of gpus to distribute  model training on. 0 for cpu')
    parser.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    parser.add_argument('--epoch-batch-progress', type=int, default=5,
                        help='number of times to log progress per epoch')

    # Network architecture
    group = parser.add_argument_group('Network architecture')
    group.add_argument('--embed-size', type=int,
                       help='number of cols in character lookup table')
    group.add_argument('--temp-conv-filters', type=int,
                       help='number of filters is doubled through each pooling step')
    group.add_argument('--blocks', nargs='+', type=int,
                       help='list of number of blocks between pooling steps')
    group.add_argument('--fc-size', type=int,
                       help='neurons in fully connected layers')
    # Regularization
    group = parser.add_argument_group('Regularization arguments')
    group.add_argument('--dropout', type=float,
                       help='dropout probability for fully connected layers')
    group.add_argument('--l2', type=float,
                       help='weight regularization penalty')

    # Optimizer
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--epochs', type=int,
                       help='num of times to loop through training data')
    group.add_argument('--min-lr', type=float,
                       help='min learning rate in cycle')
    group.add_argument('--max-lr', type=float,
                       help='max learning rate in cycle')
    group.add_argument('--lr-cycle-epochs', type=float,
                       help='number of epochs to increase then decrease learning rate over')
    group.add_argument('--lr-increase-fraction', type=float,
                       help='ratio between lr increase & decrease rates')
    group.add_argument('--momentum', type=float,
                       help='optimizer momentum')
    group.add_argument('--batch-size', type=int,
                       help='number of training examples per batch')
    parser.add_argument('--num-buckets', type=int,
                        help='num of different allowed sequence lengths')

    args = parser.parse_args()
    hyp = {k: v for k, v in vars(args).items() if v is not None}
    train(hyperparameters=hyp, channel_input_dirs={'train': args.train, 'val': args.val}, num_gpus=args.gpus)
