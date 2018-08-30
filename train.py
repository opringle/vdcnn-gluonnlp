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


def evaluate_accuracy(data_iterator, net):
    """

    :param data_iterator:
    :param net:
    :return:
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """

    :param hyperparameters:
    :param channel_input_dirs:
    :param num_gpus:
    :param kwargs:
    :return:
    """
    logging.info("Reading in data")
    train_df = pd.read_pickle(channel_input_dirs['train'])[:1000]
    val_df = pd.read_pickle(channel_input_dirs['val'])

    alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    batch_size = hyperparameters.get('batch_size', 128)

    logging.info("Building data loaders")
    train_iter, val_iter = build_dataloaders(train_df=train_df,
                                             val_df=val_df,
                                             alphabet=alph,
                                             max_utt_chars=hyperparameters.get('sequence_len', 1014),
                                             batch_size=batch_size,
                                             num_workers=multiprocessing.cpu_count())

    logging.info("Defining network architecture")
    net = CnnTextClassifier(vocab_size=len(alph),
                            embed_size=hyperparameters.get('embed_size', 16),
                            dropout=hyperparameters.get('dropout', 0.02),
                            num_label=len(train_df.intent.unique()),
                            filters=hyperparameters.get('filters', [64, 128, 256, 512]),
                            blocks=hyperparameters.get('blocks', [1, 1, 1, 1]))

    # convert network from imperitive to symbolic for increased training speed
    if hyperparameters.get('hybridize', True):
        logging.info("Hybridizing network")
        net.hybridize()

    # initialize weights depending on layer type
    logging.info("Initializing network parameters")
    net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=ctx)

    sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                              optimizer_params={'learning_rate': hyperparameters.get('learning_rate', 0.1),
                                                'momentum': hyperparameters.get('momentum', 0.9)})

    logging.info("Training")
    updates_per_epoch = train_df.shape[0] // batch_size
    accuracies = []
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
            if weight_updates % (updates_per_epoch // hyperparameters.get('epoch_batch_progress', 5)) == 0:
                logging.info("Epoch {}: Batches complete {}/{}".format(e, weight_updates, updates_per_epoch))
        train_accuracy = evaluate_accuracy(train_iter, net)
        val_accuracy = evaluate_accuracy(val_iter, net)
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
    parser.add_argument('--hybridize', type=bool,
                        help='use symbolic network graph for increased computational eff')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    parser.add_argument('--epoch-batch-progress', type=int, default=5,
                        help='number of times to log progress per epoch')

    # Network architecture
    group = parser.add_argument_group('Network architecture')
    group.add_argument('--sequence-length', type=int,
                       help='number of characters per utterance')
    group.add_argument('--num-filters', type=int,
                       help='number of filters per conv layer')
    group.add_argument('--fully-connected', type=int,
                       help='neurons in fully connected layers')

    # Regularization
    group = parser.add_argument_group('Regularization arguments')
    group.add_argument('--dropout', type=float,
                       help='dropout probability for fully connected layers')

    # Optimizer
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--epochs', type=int,
                       help='num of times to loop through training data')
    group.add_argument('--learning-rate', type=float,
                       help='optimizer learning rate')
    group.add_argument('--momentum', type=float,
                       help='optimizer momentum')
    group.add_argument('--batch-size', type=int,
                       help='number of training examples per batch')

    args = parser.parse_args()
    hyp = {k: v for k, v in vars(args).items() if v is not None}
    train(hyperparameters=hyp, channel_input_dirs={'train': args.train, 'val': args.val}, num_gpus=args.gpus)
