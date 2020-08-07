"""This file implements the pipeline for training the model.

This file defines a complete pipline to train to train and test
a transformer-based model to predict intents and slots on the TOP
dataset. Some of the code was in this file was taken
from https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    python train_model.py --dataset='atis'
"""

import os
import argparse
import json
import time
import collections
import functools

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras

from model_layers import Encoder, Decoder, OutputHead
from data_utils import load_data, create_vocabulary, load_vocabulary, create_masks
from model_utils import evaluate, MaskedLoss, IntentSlotAccuracy, build_personalize_fn, evaluate_fn
from generate_splits import generate_iid_splits, generate_non_iid_splits

tf.compat.v1.enable_v2_behavior()

BUFFER_SIZE = 1000
np.random.seed(123)

# List of optimizers currently supported.
_SUPPORTED_OPTIMIZERS = {
    'sgd': tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
    'adagrad': tf.keras.optimizers.Adagrad
}


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
    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help='The number of transformer layers.')
    parser.add_argument('--d_model',
                        type=int,
                        default=128,
                        help='The dimensionality of the embeddings.')
    parser.add_argument('--dff',
                        type=int,
                        default=512,
                        help='The hidden layer size.')
    parser.add_argument('--num_heads',
                        type=int,
                        default=4,
                        help='The number of heads in attention layer.')
    parser.add_argument('--rate',
                        type=float,
                        default=0.0,
                        help='The dropout rate to be used.')
    parser.add_argument('--dataset',
                        type=str,
                        default='top',
                        help='Type atis or snips')
    parser.add_argument('--max_input_seq_len',
                        type=int,
                        default=56,
                        help='Maximum sequence length')
    parser.add_argument('--max_output_seq_len',
                        type=int,
                        default=66,
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
                        default='inputs.txt',
                        help='Input file name.')
    parser.add_argument('--output_file',
                        type=str,
                        default='outputs.txt',
                        help='Outputs file name.')
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
        default=1,
        help='Number of epochs per round of federated training.')
    parser.add_argument('--num_rounds',
                        type=int,
                        default=1000,
                        help='Number of rounds of federated training.')
    parser.add_argument(
        '--split_type',
        type=str,
        default='non_iid',
        help='IID or non-IID splits to be used for the simulation.')
    parser.add_argument(
        '--num_clients',
        type=int,
        default=100,
        help=
        'Number of clients to be used for federated simulation for iid-splits.'
    )
    parser.add_argument('--server_optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer to be used for server training.')
    parser.add_argument('--client_optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer to be used for client training.')
    parser.add_argument('--server_lr',
                        type=float,
                        default=10.0,
                        help='Learning rate of the server optimizer.')
    parser.add_argument('--client_lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the client optimizer.')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum to be used with sgd')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.9,
                        help='Beta1 paramerter of Yogi and Adam')
    parser.add_argument('--beta2',
                        type=float,
                        default=0.999,
                        help='Beta2 paramerter of Yogi and Adam')
    parser.add_argument(
        '--clients_per_round',
        type=int,
        default=-1,
        help=
        'Number of clients for each round update. (-1 indicates all clients)')
    parser.add_argument(
        '--personalization',
        type=int,
        default=1,
        help=
        'A value of 1 indicates personalization and 0 indicates no personalization.'
    )
    parser.add_argument(
        '--pre_train_ratio',
        type=float,
        default=0.1,
        help='The fraction of the training set to be used for pre-traning.')
    parser.add_argument(
        '--p13n_ratio',
        type=float,
        default=0.8,
        help='The fraction of the training set to be used for personalization.'
    )

    arg = parser.parse_args()

    return arg


def create_load_vocab(arg,
                      file_name,
                      out_file_name,
                      pad=True,
                      unk=True,
                      sos_eos=False):
    """Creates and loads the vocab file for a given corpus.

    Args:
    arg: The output of the parser.
    file_name: The name of the file containing the corpus.
    out_file_name: The file into which the vocab should be written into.
    pad: A boolean to indicate if the pad token should be included
        in the vocabulary.
    unk: A boolean to indicate if the unknown token should be included
        in the vocabulary.
    sos_eos: A boolean to indicate if the SOS and EOS token should be included
        in the vocabulary.

    Returns:
    A dictionary of the vocabulary and it's corresponding index. It also
    includes a list of all the vocabulary.
    """

    full_path = os.path.join('./top_data', arg.train_data_path, file_name)
    output_path = os.path.join(arg.vocab_path, out_file_name)

    create_vocabulary(full_path, output_path, pad, unk, sos_eos)
    vocab = load_vocabulary(output_path)

    return vocab


def load_dataset(arg, data_path, in_vocab, out_vocab):
    """Returns the dataset that is loaded from the disk.

    Args:
    arg: The output of the parser.
    data_path: The path of the dataset to be loaded.
    in_vocab: The vocabulary of the input sentences.
    out_vocab: The vocabulary of output sequences.

    Returns:
    The input data, output data as numpy arrays.
    """

    full_path = os.path.join('./top_data', data_path)

    input_path = os.path.join(full_path, arg.input_file)
    output_path = os.path.join(full_path, arg.output_file)

    in_data, output_data = load_data(input_path, output_path, in_vocab,
                                     out_vocab, arg.max_input_seq_len,
                                     arg.max_output_seq_len)

    return in_data, output_data


def load_vocab(arg):
    """Creates and loads vocabulary for the input sentences,
    and annotated sentences.

    Args:
    arg: The output of the parser.

    Returns:
    The vocabulary for the inputs and outputs.
    """

    in_vocab = create_load_vocab(arg, arg.input_file, 'in_vocab')
    out_vocab = create_load_vocab(arg,
                                  arg.output_file,
                                  'out_vocab',
                                  sos_eos=True)

    return in_vocab, out_vocab


def create_keras_model(arg,
                       input_vocab_size,
                       out_vocab_size,
                       pe_input=64,
                       pe_output=128):
    """Defines and creates a keras transformer-based model
    to generate the annotates sequence for a given input.

    Args:
    arg: The output of the parser.
    input_vocab_size: The size of the input vocabulary.
    out_vocab_size: The size of the output vocabulary.
    pe_input: Maximum index of positional encodings required for the input.
    pe_output: Maximum index of positional encodings required for the output.

    Returns:
    A un-compiled keras model.
    """
    sent_input = keras.layers.Input(shape=(None, ))
    intent_slot_input = keras.layers.Input(shape=(None, ))
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
    pointer_mask = keras.layers.Input(shape=(
        1,
        None,
    ))

    # Define the layers
    encoder = Encoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      input_vocab_size, pe_input, arg.rate)

    decoder = Decoder(arg.num_layers, arg.d_model, arg.num_heads, arg.dff,
                      out_vocab_size, pe_output, arg.rate)

    output_head = OutputHead(out_vocab_size - arg.max_input_seq_len,
                             arg.d_model)

    # Define the forward pass of the model
    enc_output = encoder(sent_input, padding_mask)

    dec_output, _ = decoder(intent_slot_input, enc_output, look_ahead_mask,
                            padding_mask)

    output = output_head(enc_output, dec_output, pointer_mask)

    model = keras.Model(inputs=[
        sent_input, intent_slot_input, padding_mask, look_ahead_mask,
        pointer_mask
    ],
                        outputs=[output])

    return model


def preprocess(dataset, arg):
    """Preprocessing function fot the training data

    Args:
    dataset: The tensor containing the training data.
    arg: The output of the parser.

    Returns:
    The pre-processed dataset.
    """
    return dataset.repeat(arg.epochs_per_round).shuffle(BUFFER_SIZE).batch(
        arg.batch_size, drop_remainder=False)


def get_optimizers(arg):
    """Returns the optimizers for the server and the clients based
    on the input arguments.
    Args:
    arg: The output of the parser.
    Returns:
    The server and client optimizer.
    """
    server_opt_cls = _SUPPORTED_OPTIMIZERS.get(arg.server_optimizer)
    client_opt_cls = _SUPPORTED_OPTIMIZERS.get(arg.client_optimizer)

    if arg.server_optimizer == 'sgd':
        server_opt = lambda: server_opt_cls(learning_rate=arg.server_lr,
                                            momentum=arg.momentum)
    elif arg.server_optimizer in ['adam', 'adagrad']:
        server_opt = lambda: server_opt_cls(
            learning_rate=arg.server_lr, beta_1=arg.beta1, beta_2=arg.beta2)
    else:
        print('{} optimizer not supported.'.format(arg.server_optimizer))
        raise Exception

    client_opt = lambda: client_opt_cls(learning_rate=arg.client_lr)

    return server_opt, client_opt


def train_model(model, train_dataset, valid_dataset, out_vocab, num_epochs=20):
    """Pre-trains the model with a subset of the training set to
    simulate a data donation setup.

    Args:
    model: The model to be pre-trained.
    train_dataset: The training set.
    valid_set: The dataset on which the model will be evaluated.
    out_vocab: The output vocabulary
    num_epochs: Number of rounds the model has to be pre-trained for.
    """
    loss_objective = MaskedLoss()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    optimizer = tf.keras.optimizers.Adam()

    # training step
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, outputs):
        dec_inputs = outputs[:, :-1]
        dec_target = outputs[:, 1:]

        padding_mask, look_ahead_mask, pointer_mask = create_masks(
            inputs, dec_target)

        with tf.GradientTape() as tape:
            y_pred = model((inputs, dec_inputs, padding_mask, look_ahead_mask,
                            pointer_mask),
                           training=True)

            loss = loss_objective(dec_target, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    for epoch in range(num_epochs):
        train_loss.reset_states()

        for inp, out in train_dataset:
            train_step(inp, out)

        print('Epoch {} Loss {:.4f} '.format(epoch + 1, train_loss.result()))

        print('Validation Metrics :')
        val_acc = evaluate(model, valid_dataset, out_vocab)


def make_federated_data(client_data, client_ids, arg):
    """Generates the training data in the format required by tensorflow
    federated.

    Args:
    client_train_data: Collection of all the client datasets.
    client_ids: ID's of the clients to be used to create the dataset.
    arg: The output of the parser.

    Returns:
    A list of dataset for each client.
    """
    return [
        preprocess(client_data.create_tf_dataset_for_client(str(x)), arg)
        for x in client_ids
    ]


def create_tff_model(arg, in_vocab_size, out_vocab_size, input_spec):
    """Creates a enhanced model to be used by TFF.

    Args:
    arg: The output of the parser.
    input_vocab_size: The size of the input vocabulary.
    out_vocab_size: The size of the output vocabulary.
    input_spec: Types and shapes that the model expects.

    Returns:
    A enhanced TFF model.
    """
    keras_model = create_keras_model(arg, in_vocab_size, out_vocab_size)

    return tff.learning.from_keras_model(keras_model,
                                         input_spec=input_spec,
                                         loss=[MaskedLoss()],
                                         metrics=[IntentSlotAccuracy()])


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


def generate_splits(in_data, out_data, arg):
    """Generates either IID or non-IID splits of the training data
    depending on the user input.

    Args:
    in_data: A tensor of input queries.
    out_data: A tensor of output sequences.
    arg: The parsed arguments.

    Returns:
    The required splits of the data.
    """

    splits = None

    if arg.split_type == 'iid':
        splits = generate_iid_splits(in_data, out_data, arg.num_clients)
    elif arg.split_type == 'non_iid':
        splits, _, arg.num_clients = generate_non_iid_splits(in_data, out_data)

    return splits


def get_p13_data(train_in_data, train_out_data, valid_in_data, valid_out_data):
    """Generates non-IID splits of dataset to be used for personalization.

    Args:
    train_in_data: A subset of train input queries.
    train_out_data: A subset of train output sequences.
    valid_in_data: A subset of validation input queries.
    valid_out_data: A subset of validation output sequences.

    Returns:
    The personalization data needed for TFF.
    """
    in_data = np.concatenate((train_in_data, valid_in_data), axis=0)
    out_data = np.concatenate((train_out_data, valid_out_data), axis=0)

    train_splits, valid_splits, num_clients = generate_non_iid_splits(
        in_data,
        out_data,
        instance_types_per_client=3,
        instances_per_client=150,
        p13n_train_ratio=0.8)

    train_splits = tff.simulation.FromTensorSlicesClientData(train_splits)
    valid_splits = tff.simulation.FromTensorSlicesClientData(valid_splits)

    federated_p13n_data = []
    for client_id in range(num_clients):
        federated_p13n_data.append(
            collections.OrderedDict([
                ('train_data',
                 train_splits.create_tf_dataset_for_client(str(client_id))),
                ('test_data',
                 valid_splits.create_tf_dataset_for_client(str(client_id)))
            ]))

    return federated_p13n_data


def get_p13_eval(model_fn, evaluate_fn):
    """Defines the personalization strategies.

    Args:
    model_fn: A function to create a TFF model.
    evaluate_fn: A function to evaluate the TFF model.

    Returns:
    A dictionary of personalization strategies.
    """
    personalize_fn_dict = collections.OrderedDict()

    sgd_opt = lambda: tf.keras.optimizers.SGD(learning_rate=0.001)
    personalize_fn_dict['sgd'] = functools.partial(build_personalize_fn,
                                                   optimizer_fn=sgd_opt,
                                                   train_batch_size=8,
                                                   max_num_epochs=10,
                                                   num_epochs_per_eval=5,
                                                   test_batch_size=50)

    adam_opt = lambda: tf.keras.optimizers.Adam(learning_rate=2e-5)
    personalize_fn_dict['adam'] = functools.partial(build_personalize_fn,
                                                    optimizer_fn=adam_opt,
                                                    train_batch_size=8,
                                                    max_num_epochs=10,
                                                    num_epochs_per_eval=5,
                                                    test_batch_size=50)

    p13n_eval = tff.learning.build_personalization_eval(
        model_fn=model_fn,
        personalize_fn_dict=personalize_fn_dict,
        baseline_evaluate_fn=functools.partial(evaluate_fn, batch_size=8),
        max_num_samples=900)

    return p13n_eval


def main():
    """Runs the entire pipelone from loading the data and defining the model
    to training the model and evaluating the model.
    """

    arg = parse_arguments()

    in_vocab, out_vocab = load_vocab(arg)

    # Loading data
    train_in_data, train_out_data = load_dataset(arg, arg.train_data_path,
                                                 in_vocab, out_vocab)

    valid_in_data, valid_out_data = load_dataset(arg, arg.valid_data_path,
                                                 in_vocab, out_vocab)

    test_in_data, test_out_data = load_dataset(arg, arg.test_data_path,
                                               in_vocab, out_vocab)

    #Generating splits for pre-training, federated training and personalization evaluation
    central_idxs = np.random.choice(len(train_in_data),
                                    int(arg.pre_train_ratio *
                                        len(train_in_data)),
                                    replace=False)
    distributed_idxs = [
        idx for idx in np.arange(len(train_in_data)) if idx not in central_idxs
    ]

    central_in_data, central_out_data = tf.gather(
        train_in_data, central_idxs), tf.gather(train_out_data, central_idxs)

    # For personalization, split training set again
    if arg.personalization:
        federated_training_idxs = np.random.choice(distributed_idxs,
                                                   int(arg.p13n_ratio *
                                                       len(distributed_idxs)),
                                                   replace=False)
        p13_idxs = [
            idx for idx in np.arange(len(distributed_idxs))
            if idx not in federated_training_idxs
        ]

        validation_training_idxs = np.random.choice(len(valid_in_data),
                                                    int(arg.p13n_ratio *
                                                        len(valid_in_data)),
                                                    replace=False)
        validation_p13_idxs = [
            idx for idx in np.arange(len(valid_in_data))
            if idx not in validation_training_idxs
        ]

        p13_in_data, p13_out_data = tf.gather(train_in_data,
                                              p13_idxs), tf.gather(
                                                  train_out_data, p13_idxs)
        train_in_data, train_out_data = tf.gather(
            train_in_data,
            federated_training_idxs), tf.gather(train_out_data,
                                                federated_training_idxs)

        p13_valid_in_data, p13_valid_out_data = tf.gather(
            valid_in_data,
            validation_p13_idxs), tf.gather(valid_out_data,
                                            validation_p13_idxs)
        valid_in_data, valid_out_data = tf.gather(
            valid_in_data,
            validation_training_idxs), tf.gather(valid_out_data,
                                                 validation_training_idxs)
    else:
        train_in_data, train_out_data = tf.gather(train_in_data,
                                                  distributed_idxs), tf.gather(
                                                      train_out_data,
                                                      distributed_idxs)

    # Define the dataset to be used for pre-traning
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (central_in_data, central_out_data)).shuffle(1000)
    train_dataset = train_dataset.batch(32, drop_remainder=True)

    # Define the validation and test datasets on which the model will be evaluated.
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_in_data, valid_out_data))
    valid_dataset = valid_dataset.batch(2048, drop_remainder=False)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_in_data, test_out_data))
    test_dataset = test_dataset.batch(2048, drop_remainder=False)

    # Generate splits of data for federated simulation
    ftrain_data = generate_splits(train_in_data, train_out_data, arg)
    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    # Get personalization splits
    if arg.personalization:
        federated_p13n_data = get_p13_data(p13_in_data, p13_out_data,
                                           p13_valid_in_data,
                                           p13_valid_out_data)

    # Set the correct number of cliets per round.
    if arg.clients_per_round == -1:
        arg.clients_per_round = arg.num_clients

    # Define a non-federated model for checkpointing
    local_model = create_keras_model(arg, len(in_vocab['vocab']),
                                     len(out_vocab['vocab']))

    # Setup the checkpointing
    checkpoint_manager, summary_writer = manage_checkpoints(local_model, arg)
    summary_writer.set_as_default()

    # Pre-train the model
    train_model(local_model, train_dataset, valid_dataset, out_vocab)

    # Generate a sample dataset for the input spec
    raw_example_dataset = ftrain_data.create_tf_dataset_for_client('0')
    example_dataset = preprocess(raw_example_dataset, arg)

    server_opt, client_opt = get_optimizers(arg)

    model_fn = lambda: create_tff_model(arg, len(in_vocab[
        'vocab']), len(out_vocab['vocab']), example_dataset.element_spec)

    # Define the federated averaging process
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=client_opt,
        server_optimizer_fn=server_opt)

    if arg.personalization:
        p13n_eval = get_p13_eval(model_fn, evaluate_fn)

    server_state = iterative_process.initialize()

    # Initialize the server model with the pre-trained weights
    trainable_weights = [
        weights.numpy() for weights in local_model.trainable_weights
    ]
    server_state = tff.learning.state_with_new_model_weights(
        server_state, trainable_weights, local_model.non_trainable_weights)

    best_validation_acc = 0.0

    print('Training:')

    for round_num in range(1, arg.num_rounds):
        start = time.time()

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
        overall_accuracy = evaluate(local_model, valid_dataset, out_vocab)

        tf.summary.scalar('Train loss',
                          metrics._asdict()['loss'],
                          step=round_num)
        tf.summary.scalar('Train Intent Slot Accuracy',
                          metrics._asdict()['intent_slot_accuracy'],
                          step=round_num)

        tf.summary.scalar('Validation Intent Slot Accuracy',
                          overall_accuracy,
                          step=round_num)

        # If personalization has been enabled, print personalization metrics
        if round_num % 20 == 0 and arg.personalization:
            p13n_metrics = p13n_eval(server_state.model, federated_p13n_data)

            print('Server model metrics:')
            global_model_acc = np.array(
                p13n_metrics['baseline_metrics']['intent_slot_accuracy'])
            print('Overall accuracy : {}'.format(
                np.mean(global_model_acc).item()))

            print('Personalized model metrics (SGD):')

            personalized_model_acc = np.array(
                p13n_metrics['sgd']['final_model']['intent_slot_accuracy'])
            print('Overall accuracy : {}'.format(
                np.mean(personalized_model_acc).item()))

            print('Personalized model metrics (Adam):')

            personalized_model_acc = np.array(
                p13n_metrics['adam']['final_model']['intent_slot_accuracy'])
            print('Overall accuracy : {}'.format(
                np.mean(personalized_model_acc).item()))

        # Save the best model so far
        if overall_accuracy > best_validation_acc:
            best_validation_acc = overall_accuracy
            checkpoint_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                round_num, checkpoint_save_path))

        print('round {:2d}, metrics={}'.format(round_num, metrics))
        print('Time taken : {}'.format(time.time() - start))


if __name__ == '__main__':
    main()
