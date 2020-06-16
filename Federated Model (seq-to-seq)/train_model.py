"""This file implements the pipeline for training the model.

This file defines a complete pipline to train to train and test
a transformer-based model to predict intents and slots on the ATIS
ans Snips datasets. Some of the code was in this file was taken
from https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    python train_model.py --dataset='atis'
"""

import collections
import os
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras

from model_layers import Encoder, Decoder, SlotHead, IntentHead
from util import load_data, create_vocabulary, load_vocabulary, create_masks

tf.compat.v1.enable_v2_behavior()

BUFFER_SIZE = 10000


def parse_arguments():
    """Parses all the input arguments required to define
    and train the model.

    Returns:
    A parsed argument object.
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size.')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=100,
                        help='Max epochs to train.')
    parser.add_argument('--num_layers',
                        type=int,
                        default=3,
                        help='The number of transformer layers.')
    parser.add_argument('--d_model',
                        type=int,
                        default=64,
                        help='The dimensionality of the embeddings.')
    parser.add_argument('--dff',
                        type=int,
                        default=256,
                        help='The hidden layer size.')
    parser.add_argument('--num_heads',
                        type=int,
                        default=8,
                        help='The number of heads in attention layer.')
    parser.add_argument('--rate',
                        type=float,
                        default=0.1,
                        help='The dropout rate to be used.')
    parser.add_argument('--dataset',
                        type=str,
                        default='atis',
                        help='Type atis or snips')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=46,
                        help='Maximum sequence length')
    parser.add_argument('--model_path',
                        type=str,
                        default='./model',
                        help='Path to save model.')
    parser.add_argument('--vocab_path',
                        type=str,
                        default='./vocab',
                        help='Path to vocabulary files.')
    parser.add_argument('--train_data_path',
                        type=str,
                        default='train',
                        help='Path to training data files.')
    parser.add_argument('--test_data_path',
                        type=str,
                        default='test',
                        help='Path to testing data files.')
    parser.add_argument('--valid_data_path',
                        type=str,
                        default='valid',
                        help='Path to validation data files.')
    parser.add_argument('--input_file',
                        type=str,
                        default='seq.in',
                        help='Input file name.')
    parser.add_argument('--slot_file',
                        type=str,
                        default='seq.out',
                        help='Slot file name.')
    parser.add_argument('--intent_file',
                        type=str,
                        default='label',
                        help='Intent file name.')

    arg = parser.parse_args()

    return arg


def create_load_vocab(arg, file_name, out_file_name, no_pad=False):
    """Creates and loads the vocab file for a given corpus.

    Args:
    arg: The output of the parser.
    file_name: The name of the file containing the corpus.
    out_file_name: The file into which the vocab should be written into.
    no_pad: A boolean to indicate if the pad token should be included
        in the vocabulary.

    Returns:
    A dictionary of the vocabulary and it's corresponding index. It also
    includes a list of all the vocabulary.
    """

    full_path = os.path.join('./data', arg.dataset, arg.train_data_path,
                             file_name)
    output_path = os.path.join(arg.vocab_path, out_file_name)

    create_vocabulary(full_path, output_path, no_pad=no_pad)
    vocab = load_vocabulary(output_path)

    return vocab


def load_dataset(arg, data_path, in_vocab, slot_vocab, intent_vocab):
    """Returns the dataset that is loaded from the disk.

    Args:
    arg: The output of the parser.
    data_path: The path of the dataset to be loaded.
    in_vocab: The vocabulary of the input sentences.
    slot_vocab: The vocabulary of slot labels.
    intent_vocab: The vocabulary of intent labels.

    Returns:
    The input data, slot data and the intent data as numpy arrays.
    """

    full_path = os.path.join('./data', arg.dataset, data_path)

    input_path = os.path.join(full_path, arg.input_file)
    slot_path = os.path.join(full_path, arg.slot_file)
    intent_path = os.path.join(full_path, arg.intent_file)

    in_data, slot_data, intent_data = load_data(input_path, slot_path,
                                                intent_path, in_vocab,
                                                slot_vocab, intent_vocab,
                                                arg.max_seq_len)

    return in_data, slot_data, intent_data


def load_vocab(arg):
    """Creates and loads vocabulary for the input sentences,
    slot labels and intent labels.

    Args:
    arg: The output of the parser.

    Returns:
    The vocabulary for the input, slot and intents.
    """

    in_vocab = create_load_vocab(arg, arg.input_file, 'in_vocab')
    slot_vocab = create_load_vocab(arg, arg.slot_file, 'slot_vocab')
    intent_vocab = create_load_vocab(arg, arg.intent_file, 'intent_vocab',
                                     True)

    return in_vocab, slot_vocab, intent_vocab


def create_keras_model(arg,
                       input_vocab_size,
                       slot_vocab_size,
                       intent_vocab_size,
                       pe_max=64):

    sent_input = keras.layers.Input(shape=(arg.max_seq_len, ))
    slot_input = keras.layers.Input(shape=(arg.max_seq_len - 1, ))
    padding_mask = keras.layers.Input(shape=(
        1,
        1,
        arg.max_seq_len,
    ))
    look_ahead_mask = keras.layers.Input(shape=(
        1,
        arg.max_seq_len - 1,
        arg.max_seq_len - 1,
    ))
    intent_mask = keras.layers.Input(shape=(arg.max_seq_len, 1))

    encoder = Encoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      input_vocab_size, pe_max, arg.rate)

    decoder = Decoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      slot_vocab_size, pe_max, arg.rate)

    intent_head = IntentHead(intent_vocab_size, arg.d_model, arg.max_seq_len)

    slot_head = SlotHead(slot_vocab_size)

    enc_output = encoder(sent_input, padding_mask)

    dec_output, _ = decoder(slot_input, enc_output, look_ahead_mask,
                            padding_mask)

    intent_output = intent_head(
        enc_output, intent_mask)  # (batch_size, tar_seq_len, intent_vocab_size

    slot_output = slot_head(
        dec_output)  # (batch_size, tar_seq_len, slot_vocab_size)

    model = keras.Model(inputs=[
        sent_input, slot_input, padding_mask, look_ahead_mask, intent_mask
    ],
                        outputs=[slot_output, intent_output])

    return model


def preprocess(dataset, arg):
    return (dataset.shuffle(BUFFER_SIZE).batch(arg.batch_size,
                                               drop_remainder=True))


def make_federated_data(client_data, client_ids, arg):
    return [
        preprocess(client_data.create_tf_dataset_for_client(str(x)), arg)
        for x in client_ids
    ]


def masked_slot_loss(y_true, y_pred):
    """Defines the loss function to be used while training the model.
    The loss is defined as the sum of the intent loss and the slot loss
    per token. Masking is used to not compute the loss on the padding tokens.

    Args:
    slot_real: The ground truth for the slots.
    intent_real: The ground through for the intents.
    slot_pred: The slots predicted by the model.
    intent_pred: The intents predicted by the model.
    intent_loss_objective: The objective used to compute the intent loss.
    slot_loss_objective: The objective used to compute the slot loss.

    Returns:
    The total loss.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    slot_loss_ = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=slot_loss_.dtype)
    slot_loss_ *= mask
    slot_loss = tf.reduce_sum(slot_loss_) / tf.reduce_sum(mask)

    return slot_loss


def create_tff_model(arg, input_vocab_size, slot_vocab_size, intent_vocab_size,
                     input_spec):
    # TFF uses an `input_spec` so it knows the types and shapes
    # that your model expects.
    keras_model = create_keras_model(arg, input_vocab_size, slot_vocab_size,
                                     intent_vocab_size)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=[
            masked_slot_loss,
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ])


def main():
    """Runs the entire pipelone from loading the data and defining the model
    to training the model and evaluating the model.
    """

    arg = parse_arguments()

    in_vocab, slot_vocab, intent_vocab = load_vocab(arg)

    # Loading data
    train_in_data, train_slot_data, train_intent_data = load_dataset(
        arg, arg.train_data_path, in_vocab, slot_vocab, intent_vocab)

    # valid_in_data, valid_slot_data, valid_intent_data = load_dataset(
    #     arg, arg.valid_data_path, in_vocab, slot_vocab, intent_vocab)

    # test_in_data, test_slot_data, test_intent_data = load_dataset(
    #     arg, arg.test_data_path, in_vocab, slot_vocab, intent_vocab)

    targets = train_slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(
        train_in_data, targets)

    num_clients = 10
    ftrain_data = collections.OrderedDict()
    instance_per_client = len(train_in_data) // num_clients

    for i in range(num_clients):
        client_data = collections.OrderedDict(
            x=(train_in_data[i * instance_per_client:(i + 1) *
                             instance_per_client],
               train_slot_data[i * instance_per_client:(i + 1) *
                               instance_per_client, 1:],
               padding_masks[i * instance_per_client:(i + 1) *
                             instance_per_client],
               look_ahead_masks[i * instance_per_client:(i + 1) *
                                instance_per_client],
               intent_masks[i * instance_per_client:(i + 1) *
                            instance_per_client]),
            y=(targets[i * instance_per_client:(i + 1) * instance_per_client],
               train_intent_data[i * instance_per_client:(i + 1) *
                                 instance_per_client]))
        ftrain_data[str(i)] = client_data

    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    raw_example_dataset = ftrain_data.create_tf_dataset_for_client('1')
    example_dataset = preprocess(raw_example_dataset, arg)
    print(example_dataset.element_spec)

    ftrain_data = make_federated_data(ftrain_data, np.arange(10), arg)

    iterative_process = tff.learning.build_federated_averaging_process(
        lambda: create_tff_model(
            arg, len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), example_dataset.element_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02
                                                            ),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()

    num_rounds = 10
    for round_num in range(2, num_rounds):
        state, metrics = iterative_process.next(state, ftrain_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))


if __name__ == '__main__':
    main()
