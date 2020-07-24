"""This file defines functions to generated different splits of
   the data for federated simulations.
"""

import collections

import numpy as np
import tensorflow as tf
from data_utils import create_masks


def generate_iid_splits(in_data, slot_data, intent_data, num_clients=10):
    """Creates IID splits of the dataset by randomly partitioning the data
       into multiple splits.
    """
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(
        in_data, slot_targets)

    fed_data = collections.OrderedDict()
    instances_per_client = len(in_data) // num_clients

    shuffled_idxs = np.arange(len(in_data))
    np.random.shuffle(shuffled_idxs)

    for i in range(num_clients):
        client_idxs = shuffled_idxs[i * instances_per_client:(i + 1) *
                                    instances_per_client]
        client_data = collections.OrderedDict(x=(tf.gather(in_data,
                                                           client_idxs,
                                                           axis=0),
                                                 tf.gather(slot_inputs,
                                                           client_idxs,
                                                           axis=0),
                                                 tf.gather(padding_masks,
                                                           client_idxs,
                                                           axis=0),
                                                 tf.gather(look_ahead_masks,
                                                           client_idxs,
                                                           axis=0),
                                                 tf.gather(intent_masks,
                                                           client_idxs,
                                                           axis=0)),
                                              y=(tf.gather(slot_targets,
                                                           client_idxs,
                                                           axis=0),
                                                 tf.gather(intent_data,
                                                           client_idxs,
                                                           axis=0)))
        fed_data[str(i)] = client_data

    return fed_data


def generate_splits_type1(in_data, slot_data, intent_data, num_clients=10):
    """Creates non-IID splits of the dataset. Each intent type is distributed
    among the clients according to a random multinomial distribution.
    """
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(
        in_data, slot_targets)

    fed_data = collections.OrderedDict()

    unique_intents = np.unique(intent_data)

    client_idxs = collections.defaultdict(list)

    for intent_id in unique_intents:

        intent_client_distribution = np.random.randint(
            low=0, high=1000, size=num_clients).astype(np.float)
        intent_client_distribution /= np.sum(intent_client_distribution)

        intent_idxs = np.where(np.array(intent_data).squeeze() == intent_id)[0]

        client_idx_distribution = np.random.multinomial(
            1, intent_client_distribution, size=len(intent_idxs))
        client_idx_distribution = np.argmax(client_idx_distribution, axis=1)

        for client_id in range(num_clients):
            client_idxs[client_id] += intent_idxs[(
                client_idx_distribution == client_id)].tolist()

    for i in range(num_clients):
        client_idx = client_idxs[i]
        client_data = collections.OrderedDict(x=(tf.gather(in_data,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(slot_inputs,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(padding_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(look_ahead_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_masks,
                                                           client_idx,
                                                           axis=0)),
                                              y=(tf.gather(slot_targets,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_data,
                                                           client_idx,
                                                           axis=0)))
        fed_data[str(i)] = client_data

    return fed_data


def generate_splits_type2(in_data,
                          slot_data,
                          intent_data,
                          instance_types_per_client=1):
    """Creates non-IID splits of the dataset. Each client is given only a fixed number
    of intent types.
    """
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(
        in_data, slot_targets)

    fed_data = collections.OrderedDict()

    unique_intents = np.unique(intent_data)
    np.random.shuffle(unique_intents)

    num_clients = int(
        np.ceil(len(unique_intents) / float(instance_types_per_client)))

    client_idxs = collections.defaultdict(list)

    for client_id in range(num_clients):

        intent_ids = unique_intents[client_id *
                                    instance_types_per_client:(client_id + 1) *
                                    instance_types_per_client]

        for intent_id in intent_ids:
            intent_idxs = np.where(
                np.array(intent_data).squeeze() == intent_id)[0]
            client_idxs[client_id] += intent_idxs.tolist()

    for i in range(num_clients):
        client_idx = client_idxs[i]
        client_data = collections.OrderedDict(x=(tf.gather(in_data,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(slot_inputs,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(padding_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(look_ahead_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_masks,
                                                           client_idx,
                                                           axis=0)),
                                              y=(tf.gather(slot_targets,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_data,
                                                           client_idx,
                                                           axis=0)))
        fed_data[str(i)] = client_data

    return fed_data, num_clients


def generate_splits_type3(in_data,
                          slot_data,
                          intent_data,
                          instance_types_per_client=3,
                          clients_per_instance_type=3):
    """Creates non-IID splits of the dataset. Each client is given only a fixed number
    of intent types. This is different from type2 since in type 2 each intent type belongs
    exclusively to a certain user but in type 3 the instances having the same intent type can
    belong to different users.
    """
    slot_inputs, slot_targets = slot_data[:, :-1], slot_data[:, 1:]
    padding_masks, look_ahead_masks, intent_masks = create_masks(
        in_data, slot_targets)

    fed_data = collections.OrderedDict()

    unique_intents = np.unique(intent_data)
    np.random.shuffle(unique_intents)

    num_clients = int(
        np.ceil(len(unique_intents) /
                float(instance_types_per_client))) * clients_per_instance_type

    client_list = []
    for _ in range(
            int(
                np.ceil(clients_per_instance_type * len(unique_intents) /
                        num_clients))):
        client_shuffled = np.arange(num_clients)
        np.random.shuffle(client_shuffled)
        client_list.append(client_shuffled)

    client_list = np.concatenate(client_list)
    client_idxs = collections.defaultdict(list)

    for idx, intent_id in enumerate(unique_intents):

        client_ids = client_list[idx * clients_per_instance_type:(idx + 1) *
                                 clients_per_instance_type]

        intent_client_distribution = np.random.randint(
            low=0, high=1000, size=clients_per_instance_type).astype(np.float)
        intent_client_distribution /= np.sum(intent_client_distribution)

        intent_idxs = np.where(np.array(intent_data).squeeze() == intent_id)[0]

        client_idx_distribution = np.random.multinomial(
            1, intent_client_distribution, size=len(intent_idxs))
        client_idx_distribution = np.argmax(client_idx_distribution, axis=1)

        for i, client_id in enumerate(client_ids):
            client_idxs[client_id] += intent_idxs[(
                client_idx_distribution == i)].tolist()

    for i in range(num_clients):
        client_idx = client_idxs[i]
        client_data = collections.OrderedDict(x=(tf.gather(in_data,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(slot_inputs,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(padding_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(look_ahead_masks,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_masks,
                                                           client_idx,
                                                           axis=0)),
                                              y=(tf.gather(slot_targets,
                                                           client_idx,
                                                           axis=0),
                                                 tf.gather(intent_data,
                                                           client_idx,
                                                           axis=0)))
        fed_data[str(i)] = client_data

    return fed_data, num_clients
