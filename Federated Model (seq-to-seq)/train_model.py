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
from util import load_data, create_vocabulary, load_vocabulary, create_masks, evaluate
from generate_splits import generate_iid_splits, generate_splits_type1, generate_splits_type2, generate_splits_type3

tf.compat.v1.enable_v2_behavior()

BUFFER_SIZE = 1000
np.random.seed(111)


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
    parser.add_argument('--logdir',
                        type=str,
                        default='./logs/scalars/',
                        help='Directory to save the scalar logs into.')  
    parser.add_argument('--num_rounds',
                    type=int,
                    default=500,
                    help='Number of rounds of federated training.')
    parser.add_argument('--num_clients',
                    type=int,
                    default=10,
                    help='Number of clients to be used for federated simulation.')                        
    parser.add_argument('--server_lr',
                        type=float,
                        default=1.0,
                        help='Learning rate of the server optimizer.')  
    parser.add_argument('--client_lr',
                        type=float,
                        default=0.02,
                        help='Learning rate of the client optimizer.')  

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
                       pe_max=64,
                       training=True):
    """Instantiates a keras model for the task and returns it.
    """
    sent_input = keras.layers.Input(shape=(None,))
    slot_input =  keras.layers.Input(shape=(None,))
    padding_mask = keras.layers.Input(shape=(1, 1, None,))
    look_ahead_mask = keras.layers.Input(shape=(1, None, None,))
    intent_mask = keras.layers.Input(shape=(None,1))
    

    encoder = Encoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      input_vocab_size, pe_max, arg.rate, training)

    decoder = Decoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      slot_vocab_size, pe_max, arg.rate, training)

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
    """Shuffles and batches the dataset.
    """
    return (dataset.shuffle(BUFFER_SIZE).batch(arg.batch_size,
                                               drop_remainder=False))


def make_federated_data(client_data, client_ids, arg):
    """Genrates the required federated input.
    """    
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
    loss_objective = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    slot_losses = loss_objective(y_true, y_pred)

    mask = tf.cast(mask, dtype=slot_losses.dtype)
    slot_losses *= mask
    slot_loss = tf.reduce_sum(slot_losses) / tf.reduce_sum(mask)

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


def manage_checkpoints(model, path, logdir):
    """Defines the checkpoint manager and loads the latest
    checkpoint if it exists.
    Args:
    model: An instace of the tensorflow model.
    path: The path in which the checkpoint has to be saved.
    Returns:
    The checkpoint manager which is used to save the model.
    """

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    path,
                                                    max_to_keep=None)

    summary_writer = tf.summary.create_file_writer(logdir)

    return checkpoint_manager, summary_writer

def eval(state, local_model, valid_dataset, slot_vocab):

  tff.learning.assign_weights_to_keras_model(local_model, state.model)
  return evaluate(local_model, valid_dataset, slot_vocab)

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

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_in_data, valid_slot_data, valid_intent_data))
    valid_dataset = valid_dataset.batch(512, drop_remainder=False)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_in_data, test_slot_data, test_intent_data))
    test_dataset = test_dataset.batch(512, drop_remainder=False)  


    ftrain_data = generate_iid_splits(train_in_data, train_slot_data, train_intent_data, arg.num_clients)

    local_model = create_keras_model(arg,  len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), training=False)
    
    checkpoint_manager, summary_writer = manage_checkpoints(
        local_model, os.path.join('./checkpoints/iid_sgd'), arg.logdir)

    summary_writer.set_as_default()

    raw_example_dataset = ftrain_data.create_tf_dataset_for_client('1')
    example_dataset = preprocess(raw_example_dataset, arg)

    iterative_process = tff.learning.build_federated_averaging_process(
        lambda: create_tff_model(
            arg, len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), example_dataset.element_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=arg.client_lr
                                                            ),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=arg.server_lr))

    state = iterative_process.initialize()

    best_validation_acc = 0.0

    for round_num in range(1, arg.num_rounds):
        # Sample a subset of clients to be used for this round
        # client_subset = np.random.choice(num_clients, num_clients, replace=False)

        ftrain_data_subset = make_federated_data(ftrain_data, np.arange(arg.num_clients), arg)

        state, metrics = iterative_process.next(state, ftrain_data_subset)
        semantic_acc, intent_acc, f1_score = eval(state, local_model, valid_dataset, slot_vocab)

        tf.summary.scalar('Train loss', metrics._asdict()['loss'], step=round_num)
        tf.summary.scalar('Validation Semantic Accuracy', semantic_acc, step=round_num)
        tf.summary.scalar('Validation f1 Score', f1_score, step=round_num)
        tf.summary.scalar('Validation Intent Accuracy', intent_acc, step=round_num)

        if semantic_acc > best_validation_acc:
            best_validation_acc = semantic_acc
            checkpoint_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                round_num, checkpoint_save_path))

        print('round {:2d}, metrics={}'.format(round_num, metrics))


if __name__ == '__main__':
    main()
