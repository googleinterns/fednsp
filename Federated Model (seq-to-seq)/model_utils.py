"""This file defines the utils for the model traning.

This file contains functions to compute the loss
and metrics to bes used while training including
functions to compute the F1 score and
sentence-level semantic frame accuracy, etc.

Some of the code was in this file was taken from
https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU/blob/master/utils.py
"""

import tensorflow as tf
import numpy as np

from data_utils import create_masks

SOS_TOKEN = '__SOS'


def masked_slot_loss(y_true, y_pred):
    """Defines the loss function to be used while training the model.
    Masking is used to not compute the loss on the padding tokens.

    Args:
        y_true: The ground truth for the annotations.
        y_pred: The annotations predicted by the model.

    Returns:
    The total loss per output token.
    """
    loss_objective = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    slot_losses = loss_objective(y_true, y_pred)

    mask = tf.cast(mask, dtype=slot_losses.dtype)
    slot_losses *= mask
    slot_loss = tf.reduce_sum(slot_losses) / tf.reduce_sum(mask)

    return slot_loss


class IntentAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    """Class defines the intent accuracy metric to be used
    for the given task.
    """
    def __init__(self, name='intent_accuracy', dtype=tf.float32):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[1], y_pred[1], sample_weight)


class IntentSlotAccuracy(tf.keras.metrics.Metric):
    """Class defines the intent + slot accuracy metric to be used
    for the given task.
    """
    def __init__(self, name='intent_slot_accuracy', **kwargs):
        super(IntentSlotAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        slot_mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true[0], 0)),
                            tf.int32)
        slot_tp = tf.cast(
            tf.math.equal(
                y_true[0],
                tf.math.argmax(y_pred[0], axis=-1, output_type=tf.int32)),
            tf.int32)

        slot_accuracy = tf.math.reduce_sum(tf.math.multiply(
            slot_tp, slot_mask),
                                           axis=-1)
        slot_accuracy = tf.math.equal(slot_accuracy,
                                      tf.math.reduce_sum(slot_mask, axis=-1))

        intent_tp = tf.math.equal(
            y_true[1][:, 0],
            tf.math.argmax(y_pred[1], axis=-1, output_type=tf.int32))
        tp = tf.reduce_sum(
            tf.cast(tf.math.logical_and(slot_accuracy, intent_tp), tf.float32))

        total = tf.reduce_sum(tf.ones_like(y_true[1], dtype=tf.float32))

        self.true_positives.assign_add(tp)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide(self.true_positives, self.total)

    def reset_states(self):
        self.true_positives.assign(0)
        self.total.assign(0)


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


def compute_precision_recall(correct_chunk_cnt, found_pred_cnt,
                             found_correct_cnt):
    """Computes and returns the precision and recall.

    Args:
    correct_chunk_cnt: The count of correctly predicted chunks.
    found_pred_cnt:  The count of predicted chunks.
    found_correct_cnt : The actual count of chunks.

    Returns:
    The slot precision and recall
    """

    if found_pred_cnt > 0:
        precision = 100 * correct_chunk_cnt / found_pred_cnt
    else:
        precision = 0

    if found_correct_cnt > 0:
        recall = 100 * correct_chunk_cnt / found_correct_cnt
    else:
        recall = 0

    return precision, recall


def compute_start_end_chunks(last_tag, current_tag, last_type, current_type):
    """Computes if the current token is the beginning and the end of a 
    argument

    Args:
    last_tag: The previous slot tag.
    current_tag: The current slot tag.
    last_type:  The type of the previous slot tag.
    current_type:  The type of the current slot tag.

    Returns:
    Boolean variable to indicate if beginning and end of chunk.
    """
    correct_start_of_chunk = start_of_chunk(last_tag, current_tag, last_type,
                                            current_type)

    correct_end_of_chunk = end_of_chunk(last_tag, current_tag, last_type,
                                        current_type)

    return correct_start_of_chunk, correct_end_of_chunk


def compute_f1(correct_slots, pred_slots):
    """Computes and returns the f1 score of the predicted slots.

    Args:
    correct_slots: The ground truth slot labels.
    pred_slots:  The predicted slot labels.

    Returns:
    The slot f1 score.
    """

    correct_chunk_cnt = 0
    found_correct_cnt = 0
    found_pred_cnt = 0

    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        in_correct = False
        last_correct_tag = 'O'
        last_correct_type = ''
        last_pred_tag = 'O'
        last_pred_type = ''

        for c, p in zip(correct_slot, pred_slot):
            correct_tag, correct_type = split_tag_type(c)
            pred_tag, pred_type = split_tag_type(p)

            # Check if the current chunk in pred and ground truth start or end
            correct_start_of_chunk, correct_end_of_chunk = compute_start_end_chunks(
                last_correct_tag, correct_tag, last_correct_type, correct_type)

            pred_start_of_chunk, pred_end_of_chunk = compute_start_end_chunks(
                last_pred_tag, pred_tag, last_pred_type, pred_type)

            if in_correct:
                # If both chunks end, increment corrent count by 1
                if correct_end_of_chunk and pred_end_of_chunk and \
                        (last_correct_type == last_pred_type):
                    in_correct = False
                    correct_chunk_cnt += 1

                elif correct_end_of_chunk != pred_end_of_chunk or \
                                     (correct_type != pred_type):
                    in_correct = False

            # If both the prediction and ground truth chunk start
            if correct_start_of_chunk and pred_start_of_chunk and \
                    (correct_type == pred_type):
                in_correct = True

            # Increment number of chunks in ground truth if new chunk starts
            if correct_start_of_chunk:
                found_correct_cnt += 1

            # Increment number of chunks in preds if new chunk starts
            if pred_start_of_chunk:
                found_pred_cnt += 1

            last_correct_tag = correct_tag
            last_correct_type = correct_type
            last_pred_tag = pred_tag
            last_pred_type = pred_type

        if in_correct:
            correct_chunk_cnt += 1

    precision, recall = compute_precision_recall(correct_chunk_cnt,
                                                 found_pred_cnt,
                                                 found_correct_cnt)

    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score, precision, recall


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
                break

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


def evaluate(model, dataset, slot_vocab, max_len=48):
    """Evaluates the performance of the model on the given dataset and
    prints out the metrics.

    Args:
    model: The model to be evaluated.
    dataset: The dataset on which the model is to be evaluated.
    slot_vocab: The vocabulary of slot labels.
    max_len: The number of outputs to be generated by the decoder.
    """

    pred_intents = []
    pred_slots = []
    gt_intents = []
    gt_slots = []

    for inputs, slots, intents in dataset:

        decoder_input = [slot_vocab['vocab'][SOS_TOKEN]] * inputs.shape[0]
        output = tf.expand_dims(decoder_input, 1)

        for i in range(max_len - 1):
            padding_mask, look_ahead_mask, intent_mask = create_masks(
                inputs, output)

            predictions, p_intent = model(
                (inputs, output, padding_mask, look_ahead_mask, intent_mask))

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output = tf.concat([output, predicted_id], axis=-1)

            if i == max_len - 2:
                pred_intents.append(tf.argmax(p_intent, axis=-1).numpy())

        pred_slots.append(output.numpy())
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

    return semantic_acc, intent_acc, f1_score
