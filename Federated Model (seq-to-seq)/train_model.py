"""This file implements the pipeline for training the model.

This file defines a complete pipline to train to train and test
a transformer-based model to predict intents and slots on the ATIS
ans Snips datasets. Some of the code was in this file was taken
from https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    python train_model.py --dataset='atis'
"""

import os
import argparse
import json

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras

from model_layers import Encoder, Decoder, SlotHead, IntentHead
from data_utils import load_data, create_vocabulary, load_vocabulary
from model_utils import evaluate, masked_slot_loss, IntentSlotAccuracy, IntentAccuracy
from generate_splits import generate_iid_splits, generate_splits_type3

tf.compat.v1.enable_v2_behavior()

BUFFER_SIZE = 1000
np.random.seed(123)


def parse_arguments():
    """Parses all the input arguments required to define
    and train the model.

    Returns:
    A parsed argument object.
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size.')
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
                        default=4,
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
                        default=48,
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
    parser.add_argument('--chkdir',
                        type=str,
                        default='./checkpoints/reduce_num_rounds_lr',
                        help='Directory to save the checkpoints.')
    parser.add_argument('--logdir',
                        type=str,
                        default='./logs/scalars/',
                        help='Directory to save the scalar logs into.')
    parser.add_argument(
        '--epochs_per_round',
        type=int,
        default=5,
        help='Number of epochs per round of federated training.')
    parser.add_argument('--num_rounds',
                        type=int,
                        default=1000,
                        help='Number of rounds of federated training.')
    parser.add_argument(
        '--split_type',
        type=str,
        default='iid',
        help='IID or non-IID splits to be used for the simulation.')
    parser.add_argument(
        '--num_clients',
        type=int,
        default=30,
        help=
        'Number of clients to be used for federated simulation for iid-splits.'
    )
    parser.add_argument('--server_lr',
                        type=float,
                        default=1.0,
                        help='Learning rate of the server optimizer.')
    parser.add_argument('--client_lr',
                        type=float,
                        default=0.02,
                        help='Learning rate of the client optimizer.')
    parser.add_argument(
        '--clients_per_round',
        type=int,
        default=-1,
        help=
        'Number of clients for each round update. (-1 indicates all clients)')

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
    """Defines and creates a keras transformer model for the
    joint prediction of intents and slots.

    Args:
    arg: The output of the parser.
    input_vocab_size: The size of the input vocabulary.
    slot_vocab_size: The size of the slot vocabulary.
    intent_vocab_size: The size of the intent vocabulary.
    pe_max: Maximum index of positional encodings required.

    Returns:
    A un-compiled keras model.
    """
    sent_input = keras.layers.Input(shape=(None, ))
    slot_input = keras.layers.Input(shape=(None, ))
    padding_mask = keras.layers.Input(shape=(
        1,
        1,
        None,
    ))
    look_ahead_mask = keras.layers.Input(shape=(
        1,
        None,
        None,
    ))
    intent_mask = keras.layers.Input(shape=(None, 1))

    # Define the layers
    encoder = Encoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      input_vocab_size, pe_max, arg.rate)

    decoder = Decoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      slot_vocab_size, pe_max, arg.rate)

    intent_head = IntentHead(intent_vocab_size, arg.d_model, arg.max_seq_len)

    slot_head = SlotHead(slot_vocab_size)

    # Define the forward pass of the model
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
    """Preprocessing function fot the training data

    Args:
    dataset: The tensor containing the training data.
    arg: The output of the parser.
    epochs_per_rounds: Number of epochs per rounds of federated training.

    Returns:
    The pre-processed dataset.
    """
    return (dataset.repeat(arg.epochs_per_round).shuffle(BUFFER_SIZE).batch(
        arg.batch_size, drop_remainder=False))


def make_federated_data(client_data, client_ids, arg):
    """Generates the training data in the format required by tensorflow
    federated.

    Args:
    client_train_data: Collection of all the client datasets.
    client_ids: ID's of the clients to be used to create the dataset.
    epochs_per_rounds: Number of epochs per rounds of federated training.
    arg: The output of the parser.

    Returns:
    A list of dataset for each client.
    """
    return [
        preprocess(client_data.create_tf_dataset_for_client(str(x)), arg)
        for x in client_ids
    ]


def create_tff_model(arg, input_vocab_size, slot_vocab_size, intent_vocab_size,
                     input_spec):
    """Creates a enhanced model to be used by TFF.

    Args:
    arg: The output of the parser.
    input_vocab_size: The size of the input vocabulary.
    slot_vocab_size: The size of the slot vocabulary.
    intent_vocab_size: The size of the intent vocabulary.
    input_spec: Types and shapes that the model expects.

    Returns:
    A list of dataset for each client.
    """
    keras_model = create_keras_model(arg, input_vocab_size, slot_vocab_size,
                                     intent_vocab_size)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=[
            masked_slot_loss,
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ],
        metrics=[IntentSlotAccuracy(), IntentAccuracy()])


def manage_checkpoints(model, arg):
    """Defines the checkpoint manager and loads the latest
    checkpoint if it exists.

    Args:
    model: An instace of the tensorflow model.
    arg: The parsed arguments.

    Returns:
    The checkpoint manager which is used to save the model along with the
    summary writer to be used to log the metrics during training.
    """
    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    arg.chkdir,
                                                    max_to_keep=None)

    summary_writer = tf.summary.create_file_writer(
        os.path.join(arg.chkdir, arg.logdir))

    with open('config.json', 'w') as file:
        json.dump(vars(arg), file)

    return checkpoint_manager, summary_writer


def generate_splits(in_data, slot_data, intent_data, arg):
    """Generates either IID or non-IID splits of the training data
    depending on the user input.

    Args:
    in_data: A tensor of input queries.
    slot_data: A tensor of slot labels.
    intent_data: A tensor of intent labels.
    arg: The parsed arguments.

    Returns:
    The required splits of the data.
    """

    splits = None

    if arg.split_type == 'iid':
        splits = generate_iid_splits(in_data, slot_data, intent_data,
                                     arg.num_clients)
    elif arg.split_type == 'non_iid':
        splits, arg.num_clients = generate_splits_type3(
            in_data, slot_data, intent_data)

    return splits


def main():
    """Runs the entire pipelone from loading the data and defining the model
    to training the model and evaluating the model.
    """

    arg = parse_arguments()

    in_vocab, slot_vocab, intent_vocab = load_vocab(arg)

    # Loading data
    train_in_data, train_slot_data, train_intent_data = load_dataset(
        arg, arg.train_data_path, in_vocab, slot_vocab, intent_vocab)

    valid_in_data, valid_slot_data, valid_intent_data = load_dataset(
        arg, arg.valid_data_path, in_vocab, slot_vocab, intent_vocab)

    test_in_data, test_slot_data, test_intent_data = load_dataset(
        arg, arg.test_data_path, in_vocab, slot_vocab, intent_vocab)

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_in_data, valid_slot_data, valid_intent_data))
    valid_dataset = valid_dataset.batch(512, drop_remainder=False)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_in_data, test_slot_data, test_intent_data))
    test_dataset = test_dataset.batch(512, drop_remainder=False)

    # Generate splits of data for federated simulation
    ftrain_data = generate_splits(train_in_data, train_slot_data,
                                  train_intent_data, arg)
    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    if arg.clients_per_round == -1:
        arg.clients_per_round = arg.num_clients

    # Define a non-federated model for checkpointing
    local_model = create_keras_model(arg, len(in_vocab['vocab']),
                                     len(slot_vocab['vocab']),
                                     len(intent_vocab['vocab']))

    checkpoint_manager, summary_writer = manage_checkpoints(local_model, arg)

    summary_writer.set_as_default()

    # Generate a sample dataset
    raw_example_dataset = ftrain_data.create_tf_dataset_for_client('1')
    example_dataset = preprocess(raw_example_dataset, arg)

    # Define the federated averaging process
    iterative_process = tff.learning.build_federated_averaging_process(
        lambda: create_tff_model(
            arg, len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), example_dataset.element_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=arg.
                                                            client_lr),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=arg.
                                                            server_lr))

    server_state = iterative_process.initialize()

    best_validation_acc = 0.0

    for round_num in range(1, arg.num_rounds):
        # Sample a subset of clients to be used for this round
        client_subset = np.random.choice(arg.num_clients,
                                         arg.clients_per_round,
                                         replace=False)
        ftrain_data_subset = make_federated_data(ftrain_data, client_subset,
                                                 arg)

        # Perform one round of federated training
        server_state, metrics = iterative_process.next(server_state,
                                                       ftrain_data_subset)

        # Compute and log validation metrics
        tff.learning.assign_weights_to_keras_model(local_model,
                                                   server_state.model)
        semantic_acc, intent_acc, f1_score = evaluate(local_model,
                                                      valid_dataset,
                                                      slot_vocab)

        tf.summary.scalar('Train loss',
                          metrics._asdict()['loss'],
                          step=round_num)
        tf.summary.scalar('Train Intent Slot Accuracy',
                          metrics._asdict()['intent_slot_accuracy'],
                          step=round_num)

        tf.summary.scalar('Validation Intent Slot Accuracy',
                          semantic_acc,
                          step=round_num)
        tf.summary.scalar('Validation f1 Score', f1_score, step=round_num)
        tf.summary.scalar('Validation Intent Accuracy',
                          intent_acc,
                          step=round_num)

        # Save the best model so far
        if semantic_acc > best_validation_acc:
            best_validation_acc = semantic_acc
            checkpoint_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                round_num, checkpoint_save_path))

        print('round {:2d}, metrics={}'.format(round_num, metrics))


if __name__ == '__main__':
    main()
