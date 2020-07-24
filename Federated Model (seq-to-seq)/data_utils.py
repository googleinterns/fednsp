"""This file defines the utils for the model.

This file contains helper functions used by train_model.py. It defines
function for creating the vocabulary and loading and preprocessing the
dataset.

Some of the code was in this file was taken from
https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU/blob/master/utils.py
"""

import tensorflow as tf

PADDING_TOKEN = '__PAD'
UNK_TOKEN = '__UNK'
SOS_TOKEN = '__SOS'
EOS_TOKEN = '__EOS'


def create_vocabulary(input_path, output_path, no_pad=False, no_unk=False):
    """Creates the vocabulary by parsing the given data ans saves it in
    the output path.

    Args:
    input_path: The path to the corpus for which vocabulary has to be built.
    output_path: The path to which the vocabulary has to be saved.
    no_pad: A boolean variable to indicate if a padding token is to be
        added to the vocabulary.
    no_unk: A boolean variable to indicate if a unknown token is to be
        added to the vocabulary.
    """
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r') as fd, \
            open(output_path, 'w+') as out:

        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for word in words:
                if word == UNK_TOKEN:
                    pass
                if str.isdigit(word):
                    word = '0'
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        if not no_pad:
            vocab = [PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(
                vocab, key=vocab.get, reverse=True)
        else:
            vocab = [UNK_TOKEN] + sorted(vocab, key=vocab.get, reverse=True)
        for vocab_word in vocab:
            out.write(vocab_word + '\n')


def load_vocabulary(path):
    """Loads the vocabulary from the given path and constructs a dictionary for
    mapping of vocabulary to numerical ID's as well as a list for the reverse
    mapping.

    Args:
    path: The path from which the vocabulary has to be loaded.

    Returns:
    A dictionary of forward and reverse mappings.
    """

    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x, y) for (y, x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}


def sentence_to_ids(data, vocab):
    """Converts the given sentence to a list of integers based on the
    vocabulary mappings.

    Args:
    data: The sentence to be converted into a list of ID's.
    vocab: The vocabulary returned by load_vocabulary().

    Returns:
    The list of integers corresponding to the input sentence.
    """

    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for word in words:
        if str.isdigit(word):
            word = '0'
        ids.append(vocab.get(word, vocab[UNK_TOKEN]))
    return ids


def load_data(in_path,
              slot_path,
              intent_path,
              in_vocab,
              slot_vocab,
              intent_vocab,
              maxlen=48):
    """Loads the data from the given path and preprocesses the data
    by converting the tokens into ID's using the vocab dictionary.
    Additionally, tokens are padded to the maxlen before returning.

    Args:
    in_path: The path to the file contating the input queries.
    slot_path: The path to the file contating the slot labels.
    intent_path: The path to the file contating the intent labels.
    in_vocab: The vocabulary of the input sentences.
    slot_vocab: The vocabulary of slot labels.
    intent_vocab: The vocabulary of intent labels.

    Returns:
    The preprocesses input data, slot lables and the intents.
    """

    in_data = []
    slot_data = []
    intent_data = []

    with open(in_path, 'r') as input_fd, \
      open(intent_path, 'r') as intent_fd, \
      open(slot_path, 'r') as slot_fd:

        for inputs, intent, slot in zip(input_fd, intent_fd, slot_fd):
            inputs, intent, slot = inputs.rstrip(), intent.rstrip(
            ), slot.rstrip()
            in_data.append(
                sentence_to_ids(SOS_TOKEN + ' ' + inputs + ' ' + EOS_TOKEN,
                                in_vocab))
            intent_data.append(sentence_to_ids(intent, intent_vocab))
            slot_data.append(
                sentence_to_ids(SOS_TOKEN + ' ' + slot + ' ' + EOS_TOKEN,
                                slot_vocab))

    in_data = tf.keras.preprocessing.sequence.pad_sequences(in_data,
                                                            padding='post',
                                                            maxlen=maxlen)
    slot_data = tf.keras.preprocessing.sequence.pad_sequences(slot_data,
                                                              padding='post',
                                                              maxlen=maxlen)
    return in_data, slot_data, intent_data


def create_padding_mask(seq):
    """Creates the paddding mask that will be used by the encoder
    for masking out the padding tokens. It also create the intent
    mask for masking out the padding tokens in the IntentHead.

    Args:
    seq: The sequence of inputs to be passed to the model.

    Returns:
    The encoder mask and the intent mask.
    """

    enc_mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    enc_mask = enc_mask[:, tf.newaxis,
                        tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    intent_mask = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
    intent_mask = intent_mask[:, :, tf.newaxis]

    return enc_mask, intent_mask


def create_look_ahead_mask(size):
    """Generates and returns the look ahead mask of a given size
    to be used while decoding.
    """

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inputs, target):
    """Creates all the necessary masks for training.

    Args:
    inputs: The sequence of inputs to be passed to the model.
    target: The slot targets to be passed to the decoder.

    Returns:
    The encoder mask and the intent mask.
    """

    # padding mask same for encoder and decoder
    padding_mask, intent_mask = create_padding_mask(inputs)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask, _ = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask, combined_mask, intent_mask
