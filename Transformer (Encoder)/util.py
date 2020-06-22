"""This file defines the utils for the model.

This file contains helper functions used by train_model.py. It defines
function for creating the vocabulary and loading and preprocessing the
dataset. Additionally, it contains functions to compute certain
metrics like F1 score, sentence-level semantic frame accuracy, etc.
Some of the code was in this file was taken from
https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU/blob/master/utils.py

"""

import tensorflow as tf
import numpy as np

PADDING_TOKEN = '__PAD'
UNK_TOKEN = '__UNK'

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
            vocab = [PADDING_TOKEN, UNK_TOKEN] + sorted(
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
        ids.append(vocab.get(word, vocab['_UNK']))
    return ids


def pad_sentence(sequence, max_length, vocab):
    """Pads the given sentence to max_length by adding padding tokens.

    Args:
    sequence: The sentence to be padded.
    max_length: The length of the sentence after padding.
    vocab:  The vocabulary returned by load_vocabulary().

    Returns:
    The sentence padded with padding token.
    """

    return sequence + [vocab['vocab']['_PAD']] * (max_length - len(sequence))


# compute f1 score is modified from conlleval.pl
def start_of_chunk(prev_tag, tag, prev_tag_type, tag_type, chunk_start=False):
    """Checks if the current tag indicates the start of a new chunk.

    Args:
    prev_tag: The tag of the previous token.
    tag: The tag of the current token.
    prev_tag_type: The tag type of the previous token.
    tag_type: The tag type of the previous token.

    Returns:
    A boolean variable indicating if the current tag is the beginning of a new
    chunk.
    """

    if prev_tag == 'B' and tag == 'B':
        chunk_start = True
    if prev_tag == 'I' and tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_tag_type != tag_type:
        chunk_start = True
    return chunk_start


def end_of_chunk(prev_tag, tag, prev_tag_type, tag_type, chunk_end=False):
    """Checks if the current tag indicates the end of a chunk.

    Args:
    prev_tag: The tag of the previous token.
    tag: The tag of the current token.
    prev_tag_type: The tag type of the previous token.
    tag_type: The tag type of the previous token.

    Returns:
    A boolean variable indicating if the current tag is the end of a
    chunk.
    """

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag == 'E' and tag == 'E':
        chunk_end = True
    if prev_tag == 'E' and tag == 'I':
        chunk_end = True
    if prev_tag == 'E' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_tag_type != tag_type:
        chunk_end = True
    return chunk_end


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


def compute_f1(correct_slots, pred_slots):
    """Computes and returns the f1 score of the predicted slots.

    Args:
    correct_slots: The ground truth slot labels.
    pred_slots:  The predicted slot labels.

    Returns:
    The slot f1 score.
    """

    correct_chunk = {}
    correct_chunk_cnt = 0
    found_correct = {}
    found_correct_cnt = 0
    found_pred = {}
    found_pred_cnt = 0
    correct_tags = 0
    token_count = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        in_correct = False
        last_correct_tag = 'O'
        last_correct_type = ''
        last_pred_tag = 'O'
        last_pred_type = ''
        for c, p in zip(correct_slot, pred_slot):
            correct_tag, correct_type = split_tag_type(c)
            pred_tag, pred_type = split_tag_type(p)

            if in_correct:
                if end_of_chunk(last_correct_tag, correct_tag,
                                last_correct_type, correct_type) and \
                                end_of_chunk(last_pred_tag, pred_tag,
                                             last_pred_type, pred_type) and \
                                    (last_correct_type == last_pred_type):
                    in_correct = False
                    correct_chunk_cnt += 1
                    if last_correct_type in correct_chunk:
                        correct_chunk[last_correct_type] += 1
                    else:
                        correct_chunk[last_correct_type] = 1
                elif end_of_chunk(last_correct_tag, correct_tag,
                                  last_correct_type, correct_type) != \
                                  end_of_chunk(last_pred_tag, pred_tag,
                                               last_pred_type, pred_type) or \
                                     (correct_type != pred_type):
                    in_correct = False

            if start_of_chunk(last_correct_tag, correct_tag,
                              last_correct_type, correct_type) and \
                                start_of_chunk(last_pred_tag, pred_tag,
                                               last_pred_type, pred_type) and \
                                    (correct_type == pred_type):
                in_correct = True

            if start_of_chunk(last_correct_tag, correct_tag, last_correct_type,
                              correct_type):
                found_correct_cnt += 1
                if correct_type in found_correct:
                    found_correct[correct_type] += 1
                else:
                    found_correct[correct_type] = 1

            if start_of_chunk(last_pred_tag, pred_tag, last_pred_type,
                              pred_type):
                found_pred_cnt += 1
                if pred_type in found_pred:
                    found_pred[pred_type] += 1
                else:
                    found_pred[pred_type] = 1

            if correct_tag == pred_tag and correct_type == pred_type:
                correct_tags += 1

            token_count += 1

            last_correct_tag = correct_tag
            last_correct_type = correct_type
            last_pred_tag = pred_tag
            last_pred_type = pred_type

        if in_correct:
            correct_chunk_cnt += 1
            if last_correct_type in correct_chunk:
                correct_chunk[last_correct_type] += 1
            else:
                correct_chunk[last_correct_type] = 1

    if found_pred_cnt > 0:
        precision = 100 * correct_chunk_cnt / found_pred_cnt
    else:
        precision = 0

    if found_correct_cnt > 0:
        recall = 100 * correct_chunk_cnt / found_correct_cnt
    else:
        recall = 0

    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score, precision, recall


def load_data(in_path,
              slot_path,
              intent_path,
              in_vocab,
              slot_vocab,
              intent_vocab,
              maxlen=35):
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
            inputs, intent, slot = inputs.rstrip(), intent.rstrip(), slot.rstrip()
            in_data.append(sentence_to_ids(inputs, in_vocab))
            intent_data.append(sentence_to_ids(intent, intent_vocab))
            slot_data.append(sentence_to_ids(slot, slot_vocab))

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


def create_masks(inputs):
    """Creates all the necessary masks for training.

    Args:
    inputs: The sequence of inputs to be passed to the model.

    Returns:
    The encoder mask and the intent mask.
    """

    # Encoder padding mask
    enc_padding_mask, intent_mask = create_padding_mask(inputs)

    return enc_padding_mask, intent_mask


def compute_semantic_acc(slot_real, intent_real, slot_pred, intent_pred):
    """Computes the semantic accuracy of the intent and slot predictions.
    (The percentage of queries for which both intents and slots were
    predicted correctly.)

    Args:
    slot_real: The ground truth for the slots.
    intent_real: The ground through for the intents.
    slot_pred: The slots predicted by the model.
    intent_pred: The intents predicted by the model.

    Returns:
    The sematic accuracy of the predictions.
    """

    semantic_acc = (intent_pred == intent_real)

    for idx, (s_pred, s_real) in enumerate(zip(slot_pred, slot_real)):

        for i in range(len(s_real)):
            if s_real[i] == 0:
                break

            if s_pred[i] != s_real[i]:
                semantic_acc[idx] = False

    semantic_acc = semantic_acc.astype(float)
    semantic_acc = np.mean(semantic_acc) * 100.0

    return semantic_acc


def compute_metrics(slot_real, intent_real, slot_pred, intent_pred,
                    slot_vocab):
    """Computes all the relevant metrics for the predictions.

    Args:
    slot_real: The ground truth for the slots.
    intent_real: The ground through for the intents.
    slot_pred: The slots predicted by the model.
    intent_pred: The intents predicted by the model.
    slot_vocab: The vocabulary of slot labels.

    Returns:
    The intent accuracy, semantic accuracy, f1, precision and recall.
    """

    slots_pred_dec = []
    slots_real_dec = []

    #decode the predictions and ground truth
    for s_pred, s_real in zip(slot_pred, slot_real):
        pred_dec = []
        real_dec = []

        for i in range(len(s_real)):
            if s_real[i] == 0:
                break

            pred_dec.append(slot_vocab['rev'][s_pred[i]])
            real_dec.append(slot_vocab['rev'][s_real[i]])

        slots_pred_dec.append(pred_dec)
        slots_real_dec.append(real_dec)

    # compute all metrics
    intent_acc = np.mean((intent_real == intent_pred).astype(np.float)) * 100.0
    f1_score, precision, recall = compute_f1(slots_real_dec, slots_pred_dec)
    semantic_acc = compute_semantic_acc(slot_real, intent_real, slot_pred,
                                        intent_pred)

    return intent_acc, semantic_acc, f1_score, precision, recall


def evaluate(model, dataset, slot_vocab):
    """Evaluates the performance of the model on the given dataset and
    prints out the metrics.

    Args:
    model: The model to be evaluated.
    dataset: The dataset on which the model is to be evaluated.
    slot_vocab: The vocabulary of slot labels.
    """

    pred_intents = []
    pred_slots = []
    gt_intents = []
    gt_slots = []

    for inputs, slots, intents in dataset:
        enc_padding_mask, slot_mask = create_masks(inputs)
        p_slot, p_intent = model(inputs, False, enc_padding_mask, slot_mask)
        pred_slots.append(tf.argmax(p_slot, axis=-1).numpy())
        pred_intents.append(tf.argmax(p_intent, axis=-1).numpy())
        gt_slots.append(slots.numpy())
        gt_intents.append(intents.numpy().squeeze())

    pred_slots = np.vstack(pred_slots)
    pred_intents = np.hstack(pred_intents)
    gt_slots = np.vstack(gt_slots)
    gt_intents = np.hstack(gt_intents)

    intent_acc, semantic_acc, f1_score, _, _ = compute_metrics(
        gt_slots, gt_intents, pred_slots, pred_intents, slot_vocab)

    print("Intent Acc {:.4f}, Semantic Acc {:.2f}, F1 score {:.2f}".format(
        intent_acc, semantic_acc, f1_score))
