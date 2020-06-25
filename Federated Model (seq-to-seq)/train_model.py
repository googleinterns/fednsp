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
    parser.add_argument('--logdir',
                        type=str,
                        default='./logs/scalars/',
                        help='Directory to save the scalar logs into.')  
    parser.add_argument('--clients_per_round',
                    type=int,
                    default=5,
                    help='NUmber of clients for each round update.')                        

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
    return (dataset.shuffle(BUFFER_SIZE).batch(arg.batch_size,
                                               drop_remainder=False))


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

def generate_iid_splits(in_data, slot_data, intent_data, num_clients=10):
  
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(in_data, slot_targets)

    ftrain_data = collections.OrderedDict()
    instances_per_client = len(in_data)//num_clients

    shuffled_idxs = np.arange(len(in_data))
    np.random.shuffle(shuffled_idxs)

    for i in range(num_clients):
      client_idxs = shuffled_idxs[i*instances_per_client:(i+1)*instances_per_client]
      client_data = collections.OrderedDict(x=(tf.gather(in_data, client_idxs, axis=0),
                                               tf.gather(slot_inputs, client_idxs, axis=0),
                                               tf.gather(padding_masks, client_idxs, axis=0),
                                               tf.gather(look_ahead_masks, client_idxs, axis=0),
                                               tf.gather(intent_masks, client_idxs, axis=0)),
                                            y=(tf.gather(slot_targets, client_idxs, axis=0),
                                               tf.gather(intent_data, client_idxs, axis=0)))
      ftrain_data[str(i)] = client_data

    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    return ftrain_data

def generate_splits_type1(in_data, slot_data, intent_data, num_clients=10):
  
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(in_data, slot_targets)

    ftrain_data = collections.OrderedDict()

    idxs = np.arange(len(in_data))
    unique_intents = np.unique(intent_data)
    
    client_idxs = collections.defaultdict(list)
    
    for intent_id in unique_intents:
        
        intent_client_distribution = np.random.randint(low=0, high=1000, size=num_clients).astype(np.float)
        intent_client_distribution /= np.sum(intent_client_distribution)
        
        intent_idxs = np.where(np.array(intent_data).squeeze() == intent_id)[0]
        
        client_idx_distribution = np.random.multinomial(1, intent_client_distribution, size=len(intent_idxs))
        client_idx_distribution = np.argmax(client_idx_distribution, axis=1)
        
        for client_id in range(num_clients):
            client_idxs[client_id] += intent_idxs[(client_idx_distribution == client_id)].tolist()

    for i in range(num_clients):
      client_idx = client_idxs[i]
      client_data = collections.OrderedDict(x=(tf.gather(in_data, client_idx, axis=0),
                                               tf.gather(slot_inputs, client_idx, axis=0),
                                               tf.gather(padding_masks, client_idx, axis=0),
                                               tf.gather(look_ahead_masks, client_idx, axis=0),
                                               tf.gather(intent_masks, client_idx, axis=0)),
                                            y=(tf.gather(slot_targets, client_idx, axis=0),
                                               tf.gather(intent_data, client_idx, axis=0)))
      ftrain_data[str(i)] = client_data

    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    return ftrain_data

def generate_splits_type2(in_data, slot_data, intent_data, instance_types_per_client=1):
  
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(in_data, slot_targets)

    ftrain_data = collections.OrderedDict()

    idxs = np.arange(len(in_data))
    unique_intents = np.unique(intent_data)
    np.random.shuffle(unique_intents)
    
    num_clients = int(np.ceil(len(unique_intents)/float(instance_types_per_client)))
    
    client_idxs = collections.defaultdict(list)
    
    for client_id in range(num_clients):
        
        intent_ids = unique_intents[client_id*instance_types_per_client:(client_id+1)*instance_types_per_client]
        
        for intent_id in intent_ids:
            intent_idxs = np.where(np.array(intent_data).squeeze() == intent_id)[0]
            client_idxs[client_id] += intent_idxs.tolist()
        
    for i in range(num_clients):
      client_idx = client_idxs[i]
      client_data = collections.OrderedDict(x=(tf.gather(in_data, client_idx, axis=0),
                                               tf.gather(slot_inputs, client_idx, axis=0),
                                               tf.gather(padding_masks, client_idx, axis=0),
                                               tf.gather(look_ahead_masks, client_idx, axis=0),
                                               tf.gather(intent_masks, client_idx, axis=0)),
                                            y=(tf.gather(slot_targets, client_idx, axis=0),
                                               tf.gather(intent_data, client_idx, axis=0)))
      ftrain_data[str(i)] = client_data

    ftrain_data = tff.simulation.FromTensorSlicesClientData(ftrain_data)

    return ftrain_data, num_clients


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

    # if a checkpoint exists, restore the latest checkpoint.
    # if checkpoint_manager.latest_checkpoint:
    #     checkpoint.restore(checkpoint_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')

    summary_writer = tf.summary.create_file_writer(logdir)

    return checkpoint_manager, summary_writer

def eval(state, local_model, valid_dataset, slot_vocab):

  # keras_model.compile(
  #     loss=[masked_slot_loss, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)])
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


    ftrain_data, num_clients = generate_splits_type2(train_in_data, train_slot_data, train_intent_data, 2)

    local_model = create_keras_model(arg,  len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), training=False)
    
    checkpoint_manager, summary_writer = manage_checkpoints(
        local_model, os.path.join('./checkpoints/type2_2'), arg.logdir)

    summary_writer.set_as_default()

    raw_example_dataset = ftrain_data.create_tf_dataset_for_client('1')
    example_dataset = preprocess(raw_example_dataset, arg)

    

    iterative_process = tff.learning.build_federated_averaging_process(
        lambda: create_tff_model(
            arg, len(in_vocab['vocab']), len(slot_vocab['vocab']),
            len(intent_vocab['vocab']), example_dataset.element_spec),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1
                                                            ),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()

    NUM_ROUNDS = 300
    best_validation_acc = 0.0

    for round_num in range(1, NUM_ROUNDS):
        # Sample a subset of clients to be used for this round
        # client_subset = np.random.choice(num_clients, num_clients, replace=False)
        # print(client_subset)
        ftrain_data_subset = make_federated_data(ftrain_data, np.arange(num_clients), arg)

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
