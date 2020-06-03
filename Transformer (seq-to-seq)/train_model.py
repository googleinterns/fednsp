"""This file implements the pipeline for training the model.

This file defines a complete pipline to train to train and test
a transformer-based model to predict intents and slots on the ATIS
ans Snips datasets. Some of the code was in this file was taken 
from https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    python train_model.py --dataset='atis' 
"""

import tensorflow as tf
import argparse
import os
import time
from model import Net
from util import load_data, evaluate, createVocabulary, loadVocabulary, create_masks

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(slot_real, intent_real, slot_pred, intent_pred, intent_loss_object, slot_loss_object):
  mask = tf.math.logical_not(tf.math.equal(slot_real, 0))
  slot_loss_ = slot_loss_object(slot_real, slot_pred)

  mask = tf.cast(mask, dtype=slot_loss_.dtype)
  slot_loss_ *= mask
  slot_loss = tf.reduce_sum(slot_loss_)/tf.reduce_sum(mask)

  intent_loss = intent_loss_object(intent_real, intent_pred)
  intent_loss = tf.reduce_mean(intent_loss)

  total_loss = slot_loss + intent_loss

  return total_loss


def main():

  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
  parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
  parser.add_argument("--num_layers", type=int, default=3, help="The number of transformer layers.")
  parser.add_argument("--d_model", type=int, default=128, help="The dimensionality of the embeddings.")
  parser.add_argument("--dff", type=int, default=512, help="The hidden layer size.")
  parser.add_argument("--num_heads", type=int, default=8, help="The number of heads in the multi-headed attention layer.")
  parser.add_argument("--dataset", type=str, default='atis', help="Type 'atis' or 'snips' ")
  parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
  parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")
  parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
  parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
  parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
  parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
  parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
  parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

  arg = parser.parse_args()

  if arg.dataset == 'atis':
    max_seq_len = 48
  elif arg.dataset == 'snips':
    max_seq_len = 37

  full_train_path = os.path.join('./data', arg.dataset, arg.train_data_path)
  full_test_path = os.path.join('./data', arg.dataset, arg.test_data_path)
  full_valid_path = os.path.join('./data', arg.dataset, arg.valid_data_path)

  createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
  createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
  createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'),
                  no_pad=True)

  in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
  slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
  intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

  # Loading data
  train_in_data, train_slot_data, train_intent_data = load_data(os.path.join(full_train_path, arg.input_file),
                                            os.path.join(full_train_path, arg.slot_file),
                                            os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                            intent_vocab, max_seq_len)
  valid_in_data, valid_slot_data, valid_intent_data = load_data(os.path.join(full_valid_path, arg.input_file),
                                            os.path.join(full_valid_path, arg.slot_file),
                                            os.path.join(full_valid_path, arg.intent_file), in_vocab, slot_vocab,
                                            intent_vocab, max_seq_len)
  test_in_data, test_slot_data, test_intent_data = load_data(os.path.join(full_test_path, arg.input_file),
                                            os.path.join(full_test_path, arg.slot_file),
                                            os.path.join(full_test_path, arg.intent_file), in_vocab, slot_vocab,
                                            intent_vocab, max_seq_len)

  print("Data loading done!")

  # Creating tf Datasets
  BUFFER_SIZE = len(train_in_data)
  BATCH_SIZE = arg.batch_size
  steps_per_epoch = len(train_in_data)//BATCH_SIZE

  train_dataset = tf.data.Dataset.from_tensor_slices((train_in_data, train_slot_data, train_intent_data)).shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

  valid_dataset = tf.data.Dataset.from_tensor_slices((valid_in_data, valid_slot_data, valid_intent_data))
  valid_dataset = valid_dataset.batch(BATCH_SIZE, drop_remainder=False)

  test_dataset = tf.data.Dataset.from_tensor_slices((test_in_data, test_slot_data, test_intent_data))
  test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)  

  # Define model
  model = Net(num_layers=arg.num_layers, d_model=arg.d_model, num_heads=arg.num_heads, dff=arg.dff, \
    input_vocab_size=len(in_vocab['vocab']), intent_vocab_size=len(intent_vocab['vocab']), slot_vocab_size=len(slot_vocab['vocab']),\
       pe_input=64, max_seq_len=max_seq_len, pe_target=64, rate=0.5)

  # Define loss
  intent_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  slot_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

  # Define optimizer and learning rate scheduler
  learning_rate = CustomSchedule(arg.d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                      epsilon=1e-9)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_intent_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_intent_accuracy')

  checkpoint_path = './checkpoints_seq_to_seq/train'

  ckpt = tf.train.Checkpoint(model=model,
                            optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

  # logdir_path = os.path.join('logs', args.dataset, datetime.now().strftime('%Y%m%d-%H%M%S'))
  # file_writer = tf.summary.create_file_writer(logdir = '/metrics')
  # file_writer.set_as_default()

  EPOCHS = arg.max_epochs

  train_step_signature = [
      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
      tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
  ]

  @tf.function(input_signature=train_step_signature)
  def train_step(inp, slots, intents):
    slots_inp = slots[:, :-1]
    slots_tar = slots[:, 1:]
    
    padding_mask, look_ahead_mask, intent_mask = create_masks(inp, slots_tar)

    with tf.GradientTape() as tape:
      slot_pred, intent_pred = model(inp, slots_inp, True, 
                                  padding_mask, look_ahead_mask,
                                  intent_mask)
      loss = loss_function(slots_tar, intents, slot_pred, intent_pred, intent_loss_object, slot_loss_object)

    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_intent_accuracy(intents, intent_pred)

  print("Starting training...") 
  # traning loop
  for epoch in range(EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    
    for (batch, (inp, slots, intents)) in enumerate(train_dataset):
      train_step(inp, slots, intents)
        
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                  ckpt_save_path))
      
    print ('Epoch {} Loss {:.4f} Intent Acc {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), train_intent_accuracy.result()))

    print('Validation Metrics :')
    evaluate(model, valid_dataset, slot_vocab)

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  # Test the model
  evaluate(model, valid_dataset, slot_vocab)  

if __name__ == '__main__':
  main()