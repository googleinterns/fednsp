"""This file defines the utils for the model.

This file contains helper functions used by train_model.py. It defines
function for creating the vocabulary and loading and preprocessing the
dataset.

Some of the code was in this file was taken from
https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU/blob/master/utils.py
"""

import collections
import random
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

def split_tag_type(tag):
    """Extracts and returns the tag and tag type.
    """

    split_tag = tag.split('-')
    if len(split_tag) > 2 or len(split_tag) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(split_tag) == 1:
        tag = split_tag[0]
        tag_type = ""
    else:
        tag = split_tag[0]
        tag_type = split_tag[1]
    return tag, tag_type

    
def create_templates(in_path,
                     slot_path,
                     intent_path,
                     in_vocab,
                     slot_vocab,
                     intent_vocab,
                     maxlen=48):
    """Creates templates of trining instances by replacing the arguments
    with palceholders.

    Args:
    in_path: The path to the file contating the input queries.
    slot_path: The path to the file contating the slot labels.
    intent_path: The path to the file contating the intent labels.
    in_vocab: The vocabulary of the input sentences.
    slot_vocab: The vocabulary of slot labels.
    intent_vocab: The vocabulary of intent labels.

    Returns:
    The generated templates along with the actual data.
    """

    input_templates = []
    input_data = []
    slot_templates = []
    slot_data = []
    intent_labels = []

    slot_instances = collections.defaultdict(set)

    with open(in_path, 'r') as input_fd, \
      open(intent_path, 'r') as intent_fd, \
      open(slot_path, 'r') as slot_fd:

        for inputs, intent, slot in zip(input_fd, intent_fd, slot_fd):
            inputs, intent, slot = inputs.rstrip(), intent.rstrip(
            ), slot.rstrip()

            input_data.append(inputs)
            slot_data.append(slot)

            template_input, template_slot = [], []
            input_tokens, slot_tokens = inputs.split(' '), slot.split(' ')

            for idx, (input_token,
                      slot_token) in enumerate(zip(input_tokens, slot_tokens)):

                tag, tag_type = split_tag_type(slot_token)
                if tag_type == '':
                    template_slot.append(slot_token)
                    template_input.append(input_token)
                else:
                    # Append to the current arguement
                    if tag == 'I':
                        slot_instance += (' ' + input_token)
                    else:
                        slot_instance = input_token
                        template_input.append(tag_type.upper())
                        template_slot.append(tag_type)

                    # In the case of argument being at the end.
                    if (idx == len(slot_tokens) -
                            1) or (not slot_tokens[idx + 1].startswith('I')):
                        slot_instances[tag_type].add(slot_instance)

            input_templates.append(' '.join(template_input))
            slot_templates.append(' '.join(template_slot))
            intent_labels.append(intent)

    return input_templates, slot_templates, input_data, slot_data, intent_labels, slot_instances


def generate_sythetic_data(input_templates, slot_templates, intent_labels,
                           num_instances_to_generate, slot_instances):
    """Genrates additional synthetic data using the given templates and the
    instances of the arguments.

    Args:
    input_templates: A list of input templates.
    slot_templates: A list of slot templates.
    intent_labels: The intent labels for the templates.
    num_instances_to_generate: Number of synthetic examples to generate.
    slot_instances: A dictionary of the instances of each argument type.

    Returns:
    The synthetic dataset.
    """

    template_idx_list = np.random.choice(len(slot_templates),
                                         num_instances_to_generate)

    syn_inputs, syn_slots, syn_intents = [], [], []
    for idx in template_idx_list:

        input_template, slot_template, intent = input_templates[
            idx], slot_templates[idx], intent_labels[idx]
        input_tokens, slot_tokens = input_template.split(
            ' '), slot_template.split(' ')

        syn_input, syn_slot = [], []
        for input_token, slot_token in zip(input_tokens, slot_tokens):
            if slot_token == 'O':
                syn_input.append(input_token)
                syn_slot.append(slot_token)
            else:
                # Randomly sample a argument isntance
                input_sample = random.sample(slot_instances[slot_token], 1)[0]
                for i, input_word in enumerate(input_sample.split(' ')):
                    syn_input.append(input_word)
                    if i == 0:
                        syn_slot.append('B-' + slot_token)
                    else:
                        syn_slot.append('I-' + slot_token)

        syn_inputs.append(' '.join(syn_input))
        syn_slots.append(' '.join(syn_slot))
        syn_intents.append(intent)

    return syn_inputs, syn_slots, syn_intents


def create_augmented_dataset(in_path,
                             slot_path,
                             intent_path,
                             in_vocab,
                             slot_vocab,
                             intent_vocab,
                             maxlen=48):
    """Augments given dataset with synthetic examples.

    Args:
    in_path: The path to the file contating the input queries.
    slot_path: The path to the file contating the slot labels.
    intent_path: The path to the file contating the intent labels.
    in_vocab: The vocabulary of the input sentences.
    slot_vocab: The vocabulary of slot labels.
    intent_vocab: The vocabulary of intent labels.

    Returns:
    The augmented dataset.
    """
    input_templates, slot_templates, input_data, slot_data, intent_labels, slot_instances = create_templates(
        in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab,
        maxlen)
    syn_input_data, syn_slot_data, syn_intent_labels = generate_sythetic_data(
        input_templates, slot_templates, intent_labels, len(input_data),
        slot_instances)

    augmented_input_data = input_data + syn_input_data
    augmented_slot_data = slot_data + syn_slot_data
    augmented_intent_data = intent_labels + syn_intent_labels

    return augmented_input_data, augmented_slot_data, augmented_intent_data


def create_augmented_tf_dataset(in_path,
                                slot_path,
                                intent_path,
                                in_vocab,
                                slot_vocab,
                                intent_vocab,
                                maxlen=48):
    """Augments given dataset with synthetic examples and returns
    the dataset after pre-processing.


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

    aug_inputs, aug_slots, aug_intents = create_augmented_dataset(
        in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab,
        maxlen)
    in_data = []
    slot_data = []
    intent_data = []

    for inputs, intent, slot in zip(aug_inputs, aug_intents, aug_slots):
        inputs, intent, slot = inputs.rstrip(), intent.rstrip(), slot.rstrip()
        in_data.append(sentence_to_ids('__SOS  ' + inputs + ' __EOS',
                                       in_vocab))
        intent_data.append(sentence_to_ids(intent, intent_vocab))
        slot_data.append(
            sentence_to_ids('__SOS  ' + slot + ' __EOS', slot_vocab))

    in_data = tf.keras.preprocessing.sequence.pad_sequences(in_data,
                                                            padding='post',
                                                            maxlen=maxlen)
    slot_data = tf.keras.preprocessing.sequence.pad_sequences(slot_data,
                                                              padding='post',
                                                              maxlen=maxlen)
    in_data = tf.cast(in_data, tf.int32)
    slot_data = tf.cast(slot_data, tf.int32)
    intent_data = tf.cast(intent_data, tf.int32)

    return in_data, slot_data, intent_data
