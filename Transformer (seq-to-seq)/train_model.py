"""This file implements the pipeline for training the model.

This file defines a complete pipline to train to train and test
a transformer-based model to predict intents and slots on the ATIS
ans Snips datasets. Some of the code was in this file was taken
from https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    python train_model.py --dataset='atis'
"""

import argparse
import os
import time
import tensorflow as tf
from model import Net
from util import load_data, evaluate, create_vocabulary, load_vocabulary, create_masks


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


def create_tf_datasets(arg, in_vocab, slot_vocab, intent_vocab):
    """Returns the tensorflow dataloader for the train, test and
    validation data.

    Args:
    arg: The output of the parser.
    in_vocab: The vocabulary of the input sentences.
    slot_vocab: The vocabulary of slot labels.
    intent_vocab: The vocabulary of intent labels.

    Returns:
    The tensorflow dataloader for the train, test and validation data.
    """

    # Loading data
    train_in_data, train_slot_data, train_intent_data = load_dataset(
        arg, arg.train_data_path, in_vocab, slot_vocab, intent_vocab)

    valid_in_data, valid_slot_data, valid_intent_data = load_dataset(
        arg, arg.valid_data_path, in_vocab, slot_vocab, intent_vocab)

    test_in_data, test_slot_data, test_intent_data = load_dataset(
        arg, arg.test_data_path, in_vocab, slot_vocab, intent_vocab)

    # Creating tf Datasets
    buffer_size = len(train_in_data)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_in_data, train_slot_data,
         train_intent_data)).shuffle(buffer_size)
    train_dataset = train_dataset.batch(arg.batch_size, drop_remainder=True)

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_in_data, valid_slot_data, valid_intent_data))
    valid_dataset = valid_dataset.batch(512, drop_remainder=False)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_in_data, test_slot_data, test_intent_data))
    test_dataset = test_dataset.batch(512, drop_remainder=False)

    return train_dataset, valid_dataset, test_dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines the learning rate scheduler to be used while training
    the model. This schedule was used in teh Attention is all you need
    paper.

    Attributes:
        d_model: The dimensionality of the contextual embeddings.
        warmup_steps: The number of steps for which the learning rate
            increases.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        lr_option_1 = tf.math.rsqrt(step)
        lr_option_2 = step * (self.warmup_steps**-1.5)

        lr_chosen = tf.math.minimum(lr_option_1, lr_option_2)
        return tf.math.rsqrt(self.d_model) * lr_chosen


def loss_function(slot_real, intent_real, slot_pred, intent_pred,
                  intent_loss_objective, slot_loss_objective):
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

    mask = tf.math.logical_not(tf.math.equal(slot_real, 0))
    slot_loss_ = slot_loss_objective(slot_real, slot_pred)

    mask = tf.cast(mask, dtype=slot_loss_.dtype)
    slot_loss_ *= mask
    slot_loss = tf.reduce_sum(slot_loss_) / tf.reduce_sum(mask)

    intent_loss = intent_loss_objective(intent_real, intent_pred)
    intent_loss = tf.reduce_mean(intent_loss)

    total_loss = slot_loss + intent_loss

    return total_loss


def manage_checkpoints(model, path, optimizer):
    """Defines the checkpoint manager and loads the latest
    checkpoint if it exists.

    Args:
    model: An instace of the tensorflow model.
    path: The path in which the checkpoint has to be saved.
    optimizer: The optimizer used to train the model.

    Returns:
    The checkpoint manager which is used to save the model.
    """

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    path,
                                                    max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return checkpoint_manager


def train_model(model, arg, train_dataset, valid_dataset, slot_vocab):
    """Defines the model training code as well the training loop.

    Args:
    arg: The output of the parser.
    train_dataset: The dataloader for the train data.
    valid_dataset: The dataloader for the validation data.
    slot_vocab: The vocabulary of slot labels which will be used during
        evaluation.

    Returns:
    The total loss.
    """

    # Define losses
    intent_loss_objective = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    slot_loss_objective = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Define optimizer and learning rate scheduler
    learning_rate = CustomSchedule(arg.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    checkpoint_manager = manage_checkpoints(
        model, os.path.join('./checkpoints', arg.dataset), optimizer)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_intent_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_intent_accuracy')

    # training step
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, slots, intents):
        slots_input = slots[:, :-1]
        slots_target = slots[:, 1:]

        padding_mask, look_ahead_mask, intent_mask = create_masks(
            inputs, slots_target)

        with tf.GradientTape() as tape:
            slot_pred, intent_pred = model(inputs, slots_input, True,
                                           padding_mask, look_ahead_mask,
                                           intent_mask)
            loss = loss_function(slots_target, intents, slot_pred, intent_pred,
                                 intent_loss_objective, slot_loss_objective)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_intent_accuracy(intents, intent_pred)

    # traning loop
    for epoch in range(arg.max_epochs):
        start = time.time()

        train_loss.reset_states()

        for inp, slots, intents in train_dataset:
            train_step(inp, slots, intents)

        if (epoch + 1) % 5 == 0:
            checkpoint_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                epoch + 1, checkpoint_save_path))

        print('Epoch {} Loss {:.4f} Intent Acc {:.4f}'.format(
            epoch + 1, train_loss.result(), train_intent_accuracy.result()))

        print('Validation Metrics :')
        evaluate(model, valid_dataset, slot_vocab)

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    return model


def main():
    """Runs the entire pipelone from loading the data and defining the model
    to training the model and evaluating the model.
    """

    arg = parse_arguments()

    in_vocab, slot_vocab, intent_vocab = load_vocab(arg)
    train_dataset, valid_dataset, test_dataset = create_tf_datasets(
        arg, in_vocab, slot_vocab, intent_vocab)

    # Define model
    model = Net(num_layers=arg.num_layers,
                d_model=arg.d_model,
                num_heads=arg.num_heads,
                dff=arg.dff,
                input_vocab_size=len(in_vocab['vocab']),
                intent_vocab_size=len(intent_vocab['vocab']),
                slot_vocab_size=len(slot_vocab['vocab']),
                pe_max=64,
                max_seq_len=arg.max_seq_len)

    model = train_model(model, arg, train_dataset, valid_dataset, slot_vocab)

    # Test the model
    evaluate(model, test_dataset, slot_vocab)


if __name__ == '__main__':
    main()
