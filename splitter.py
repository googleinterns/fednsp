"""Program to split a dataset into two parts in a given ratio.

This file contains functions required to generate two splits of a given dataset 
in a given ratio. Additionally, the new splits generated will have the same 
distribution (intents+slots) as that of the original dataset. The split is 
generated by randomly selecting a fraction of the given dataset for each intent 
separately.

  Typical usage example:

  python splitter.py --input_file_path='./data/atis/train/seq.in' --slot_file_path='./data/atis/train/seq.out' 
    --intent_file_path='./data/atis/train/label' --out_dir_split_1='./split_1' --out_dir_split_2='./split_2'
"""

import os
import argparse
import logging
import sys
import numpy as np
import math 
from collections import defaultdict


def compute_distribution(path):
  """
  Computes and displays the distribution of a given split of the dataset.

  Args:
    path : The path to the intent file of the dataset.

  A sample output generated by this function is as follows:
    The distribution is as follows:
    atis_flight : 73.8946%
    atis_airfare : 8.5976%
    atis_city : 0.4020%
    atis_flight#atis_airfare : 0.4243%

  """
  count = defaultdict(int)
  total_count = 0.0
  with open(path, 'r') as fd:
    for line in fd:
      label = line.rstrip()
      count[label] += 1
      total_count += 1
    
  print("The distribution is as follows:")
  for label in count.keys():
    print("{} : {:.4f}%".format(label, (count[label]/total_count)*100))


def group_intents(input_file, intent_file, slot_file):
  """
  Groups the dataset based on the intents and returns it.

  Args:
    input_file : The path to the input file
    intent_file : The path to the intent file
    slot_file : The path to the slot file

  Returns:
    A dict mapping intents to a list of tuples. Each tuple contains the 
    input sentence and it's corresponding slots have a given intent.
    
  """
  intent_groups = defaultdict(list)
    
  with open(input_file, 'r') as input_fd, \
       open(intent_file, 'r') as intent_fd, \
       open(slot_file, 'r') as slot_fd:
    
      for ip, intent, slot in zip(input_fd, intent_fd, slot_fd):
        ip, intent, slot = ip.rstrip(), intent.rstrip(), slot.rstrip() 
        intent_groups[intent].append((ip, slot))

  return intent_groups


def generate_split(intent_groups, split_val):
  """
  Splits the intent groups into two in a given ratio while maintaining
  the same distribution.

  Args:
    intent_groups : A dict of tuples that groups together the instances that
                    have the same intent
    split_val : The fraction of the dataset to be used as the first split

  Returns:
    Two dictionaries mapping intents to a list of tuples. The union of these
    two dicts gives the input_dict.
  """

  split_1 = defaultdict(list)
  split_2 = defaultdict(list)

  for intent, val in intent_groups.items():
    cur_size = len(val)
    new_size = math.ceil(cur_size*split_val)
    
    idx_list = np.arange(cur_size)
    np.random.shuffle(idx_list)
    
    split_idx_list_1 = idx_list[:new_size]
    split_idx_list_2 = idx_list[new_size:]
    split_1[intent] = [val[idx] for idx in split_idx_list_1]
    split_2[intent] = [val[idx] for idx in split_idx_list_2]
    
  return split_1, split_2


def save_dataset(dataset, output_dir):
  """
  Saves the split dataset to the disk at a given location.

  Args:
    dataset : A dict of tuples of the form {intent : (sentence, slots)}
    output_dir : The directory into which the split should be saved
  """

  input_file = os.path.join(output_dir, "seq.in")
  intent_file = os.path.join(output_dir, "label")
  slot_file = os.path.join(output_dir, "seq.out")
    
  with open(input_file, 'w+') as input_fd, \
       open(intent_file, 'w+') as intent_fd, \
       open(slot_file, 'w+') as slot_fd:
  
    for intent, val in dataset.items():
      for (ip, slot) in val:
        input_fd.write(ip + '\n')
        intent_fd.write(intent + '\n')        
        slot_fd.write(slot + '\n')        


def split_fraction(input_file, intent_file, slot_file, out_dir_split_1, out_dir_split_2, split_val=0.3):
  """
  This function creates two splits of the dataset which will have the same distribution (intent)
  as the original dataset.

   Args:
    input_file : The path to the input file
    intent_file : The path to the intent file
    slot_file : The path to the slot file
    out_dir_split_1 : The directory into which the first split should be saved
    out_dir_split_2 : The directory into which the second split should be saved
    split_val : The fraction of the dataset to be used as the first split
  """

  intent_groups = group_intents(input_file, intent_file, slot_file)
  split_1, split_2 = generate_split(intent_groups, split_val)
  save_dataset(split_1, out_dir_split_1)
  save_dataset(split_2, out_dir_split_2)


def main():
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--split_val", type=float, default=0.3, help="The fraction of the original dataset to be used to create the new dataset ")
  parser.add_argument("--input_file_path", type=str, default='./data/atis/train/seq.in', help="Input file path.")
  parser.add_argument("--slot_file_path", type=str, default='./data/atis/train/seq.out', help="Slot file path.")
  parser.add_argument("--intent_file_path", type=str, default='./data/atis/train/label', help="Intent file path.")
  parser.add_argument("--out_dir_split_1", type=str, default=None, help="Path to save a split of the dataset.")
  parser.add_argument("--out_dir_split_2", type=str, default=None, help="Path to save the remaining part of the dataset.")

  arg = parser.parse_args()

  if arg.out_dir_split_1 is None or arg.out_dir_split_2 is None:
    print('Output directory cannot be empty')
    exit(1)
  
  split_fraction(arg.input_file_path, arg.intent_file_path, arg.slot_file_path, \
    arg.out_dir_split_1, arg.out_dir_split_2, arg.split_val)


if __name__ == '__main__':
  main()

