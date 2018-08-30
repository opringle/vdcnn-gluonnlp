import argparse
import logging
import mxnet as mx
from mxnet import nd, gluon, autograd
from dataset import UtteranceDataset
from model import CnnTextClassifier
import pandas as pd
import multiprocessing


def build_dataloaders(train_df, val_df, alphabet, max_utt_chars, batch_size, num_workers):
    """

    :param train_df:
    :param val_df:
    :param alphabet:
    :param max_utt_chars:
    :param batch_size:
    :param num_workers:
    :return:
    """
    train_dataset = UtteranceDataset(data=train_df.utterance.values, labels=train_df.intent.values, alphabet=alphabet,
                                     feature_len=max_utt_chars)

    test_dataset = UtteranceDataset(data=val_df.utterance.values, labels=val_df.intent.values, alphabet=alphabet,
                                    feature_len=max_utt_chars)

    train_iter = gluon.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, last_batch='discard',
                                       num_workers=num_workers)

    test_iter = gluon.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, last_batch='discard',
                                      num_workers=num_workers)
    return train_iter, test_iter


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """

    :param hyperparameters:
    :param channel_input_dirs:
    :param num_gpus:
    :param kwargs:
    :return:
    """
    train_df = pd.read_pickle(channel_input_dirs['train'])
    val_df = pd.read_pickle(channel_input_dirs['val'])

    alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    batch_size = hyperparameters.get('batch_size', 100)

    train_iter, val_iter = build_dataloaders(train_df=train_df,
                                             val_df=val_df,
                                             alphabet=alph,
                                             max_utt_chars=hyperparameters.get('sequence_len', 1014),
                                             batch_size=batch_size,
                                             num_workers=multiprocessing.cpu_count())

    net = CnnTextClassifier(num_filters=hyperparameters.get('num_filters', 100),
                            fully_connected=hyperparameters.get('fully_connected', 100),
                            dropout=hyperparameters.get('dropout', 0.2),
                            num_outputs=len(train_df.intent.unique()))

    # convert network from imperitive to symbolic for increased training speed
    net.hybridize()

    # initialize weights depending on layer type
    net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=ctx)

    sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                              optimizer_params={'learning_rate': hyperparameters.get('learning_rate', 0.001),
                                                'momentum': hyperparameters.get('momentum', 0.99)})

    for e in range(hyperparameters.get('epochs', 10)):
        epoch_loss = 0
        weight_updates = 0
        for data, label in train_iter:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                pred = net(data)
                loss = sm_loss(pred, label)
            loss.backward()
            optimizer.step(hyperparameters.get('batch_size', batch_size))
            epoch_loss += nd.sum(loss).asscalar()
            weight_updates += 1
            print("Epoch {} Batches complete {}/{}".format(e, weight_updates, train_df.shape[0]//batch_size), end='\r')
        print("Epoch {} Train Loss = {:.4}".format(e, epoch_loss/weight_updates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN text classification")

    parser.add_argument('--train', type=str, required=True, help='path to pandas pickle of training data')
    parser.add_argument('--val', type=str, required=True, help='path to pandas pickle of validation data')

    parser.add_argument('--gpus', type=int, default=0, help='num of gpus to distribute  model training on. 0 for cpu')

    parser.add_argument('--epochs', type=int, help='num of times to loop through training data')

    parser.add_argument('--learning-rate', type=float, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, help='optimizer momentum')

    parser.add_argument('--sequence-length', type=int, help='number of characters per utterance')
    parser.add_argument('--batch-size', type=int, help='number of training examples per batch')

    parser.add_argument('--num-filters', type=int, help='number of filters per conv layer')
    parser.add_argument('--fully-connected', type=int, help='neurons in fully connected layers')
    parser.add_argument('--dropout', type=float, help='dropout probability for fully connected layers')

    args = parser.parse_args()
    hyp = {k: v for k, v in vars(args).items() if v is not None}
    train(hyperparameters=hyp, channel_input_dirs={'train': args.train, 'val': args.val}, num_gpus=args.gpus)
