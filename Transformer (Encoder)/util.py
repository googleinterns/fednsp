"""This file defines the utils for the model.

This file contains helper functions used by train_model.py. It defines
function for creating the vocabulary and loading and preprocessing the 
dataset. Additionally, it contains functions to compute certain
metrics like F1 score, sentence-level semantic frame accuracy, etc.
Some of the code was in this file was taken from 
https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU/blob/master/utils.py

"""

import os
import sys
import tensorflow as tf
import numpy as np
import time

def createVocabulary(input_path, output_path, no_pad=False, no_unk=False):
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

            for w in words:
                if w == '_UNK':
                    pass
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        for v in vocab:
            out.write(v + '\n')


def loadVocabulary(path):
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


def sentenceToIds(data, vocab):
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
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab['_UNK']))
    return ids


def padSentence(s, max_length, vocab):
    return s + [vocab['vocab']['_PAD']] * (max_length - len(s))


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                    __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                    (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0

    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def load_data(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab, maxlen=35):
    
  in_data = []
  slot_data = []
  intent_data = []

  with open(in_path, 'r') as input_fd, \
    open(intent_path, 'r') as intent_fd, \
    open(slot_path, 'r') as slot_fd:
            
    for ip, intent, slot in zip(input_fd, intent_fd, slot_fd):
      ip, intent, slot = ip.rstrip(), intent.rstrip(), slot.rstrip()
      in_data.append(sentenceToIds(ip, in_vocab))
      intent_data.append(sentenceToIds(intent, intent_vocab))
      slot_data.append(sentenceToIds(slot, slot_vocab))
      
  in_data = tf.keras.preprocessing.sequence.pad_sequences(in_data, padding='post', maxlen=maxlen)
  slot_data = tf.keras.preprocessing.sequence.pad_sequences(slot_data, padding='post', maxlen=maxlen)
  return in_data, slot_data, intent_data



def create_padding_mask(seq):
  enc_mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)
    
  slot_mask = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
  slot_mask = slot_mask[:, :, tf.newaxis]

  return enc_mask, slot_mask


def create_masks(inp):
  # Encoder padding mask
  enc_padding_mask, slot_mask = create_padding_mask(inp)
  
  return enc_padding_mask, slot_mask


def compute_semantic_acc(slot_real, intent_real, slot_pred, intent_pred):
    
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


def compute_metrics(slot_real, intent_real, slot_pred, intent_pred, slot_vocab):
  
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
  intent_acc = np.mean((intent_real == intent_pred).astype(np.float))*100.0
  f1, precision, recall = computeF1Score(slots_real_dec, slots_pred_dec)
  semantic_acc = compute_semantic_acc(slot_real, intent_real, slot_pred, intent_pred)
    
  return intent_acc, semantic_acc, f1, precision, recall


def evaluate(model, dataset, slot_vocab):
  
  pred_intents = []
  pred_slots = []
  gt_intents = []
  gt_slots = []

  for (batch, (inp, slots, intents)) in enumerate(dataset):
    enc_padding_mask, slot_mask = create_masks(inp)
    p_slot, p_intent = model(inp, False, 
                                 enc_padding_mask, slot_mask)
    pred_slots.append(tf.argmax(p_slot, axis=-1).numpy())
    pred_intents.append(tf.argmax(p_intent, axis=-1).numpy())
    gt_slots.append(slots.numpy())
    gt_intents.append(intents.numpy().squeeze())
    
  pred_slots = np.vstack(pred_slots)
  pred_intents = np.hstack(pred_intents)  
  gt_slots = np.vstack(gt_slots) 
  gt_intents = np.hstack(gt_intents)  
    
  intent_acc, semantic_acc, f1, _, _ = compute_metrics(gt_slots, gt_intents, pred_slots, pred_intents, slot_vocab)

  print("Intent Acc {:.4f}, Semantic Acc {:.2f}, F1 score {:.2f}".format(intent_acc, semantic_acc, f1))

