"""This file defines functions to generated different splits of
   the data for federated simulations.
"""

from collections import OrderedDict, defaultdict

import numpy as np
import tensorflow as tf
from data_utils import create_masks


def gather_indices(idxs, in_data, out_inputs, padding_masks, look_ahead_masks,
                   pointer_masks, out_targets):
    """Gathes the given indices from the given data and returns it as a OrderedDict
    """

    data = OrderedDict(x=(tf.gather(in_data, idxs,
                                    axis=0), tf.gather(out_inputs,
                                                       idxs,
                                                       axis=0),
                          tf.gather(padding_masks, idxs, axis=0),
                          tf.gather(look_ahead_masks, idxs, axis=0),
                          tf.gather(pointer_masks, idxs, axis=0)),
                       y=tf.gather(out_targets, idxs, axis=0))

    return data


def create_tff_dataset(client_idxs,
                       in_data,
                       out_data,
                       num_clients,
                       p13n_train_ratio=1.0):
    """Generates the dataset format required for tensorflow federated.

    Args:
    client_idxs: A list of clients for each client.
    in_data: The input data.
    out_data: The annotated outputs.
    num_clients: Number of clients.
    p13n_train_ratio: To be used for personalization experiments and indicates
                      the ration of train and eval splits.

    Returns:
    A dictionary of client ids mapped to the client dataset and an
    additional validation dataset in case of personalization.
    """
    train_fed_data = OrderedDict()

    # Validation dataset for personalization experiments
    valid_fed_data = OrderedDict()

    out_inputs, out_targets = out_data[:, :-1], out_data[:, 1:]
    padding_masks, look_ahead_masks, pointer_masks = create_masks(
        in_data, out_targets)

    for i in range(num_clients):
        client_idx = np.array(client_idxs[i])

        # For personalization split the instance into training and validation
        # splits.
        if p13n_train_ratio != 1.0:
            np.random.shuffle(client_idx)
            train_idxs = client_idx[:int(p13n_train_ratio * len(client_idx))]
            eval_idxs = client_idx[int(p13n_train_ratio * len(client_idx)):]
        else:
            train_idxs = client_idx

        client_data = gather_indices(train_idxs, in_data, out_inputs,
                                     padding_masks, look_ahead_masks,
                                     pointer_masks, out_targets)

        train_fed_data[str(i)] = client_data

        # Create the validation dataset for personalization
        if p13n_train_ratio != 1.0:
            client_data = gather_indices(train_idxs, in_data, out_inputs,
                                         padding_masks, look_ahead_masks,
                                         pointer_masks, out_targets)

            valid_fed_data[str(i)] = client_data

    return train_fed_data, valid_fed_data


def generate_iid_splits(in_data, out_data, num_clients=10):
    """Creates IID splits of the dataset by randomly partitioning the data
       into multiple splits.

    Args:
    in_data: The input data.
    out_data: The annotated outputs.
    num_clients: Number of clients.

    Returns:
    A dictionary of client ids mapped to the client dataset.
    """
    instances_per_client = len(in_data) // num_clients

    shuffled_idxs = np.arange(len(in_data))
    np.random.shuffle(shuffled_idxs)

    client_idxs = defaultdict(list)

    for i in range(num_clients):
        client_idxs[i] = shuffled_idxs[i * instances_per_client:(i + 1) *
                                       instances_per_client]

    fed_data = create_tff_dataset(client_idxs, in_data, out_data, num_clients)

    return fed_data


def generate_non_iid_splits(in_data,
                            out_data,
                            instance_types_per_client=3,
                            instances_per_client=250,
                            p13n_train_ratio=1.0):
    """To generate non-IID splits the dataset is partitioned based on the
    root intent. Each client is assigned training instances only from a
    subset of intent types and clients will have different subsets of intents.
    The process consists of two steps. In the first step the training instances
    are grouped according to their intent types. This is followed by generation
    of new client splits till all the training samples are used up. The
    generation of the client dataset against consists of three steps. First, a
    random subset of intent types are selected for each client. Following that,
    a random multinomial distribution over that subset of clients is generated.
    Finally, training instance belonging to the choses subset are sampled based
    on the multinomial distribution without replacement. This process is
    continued till all the training instances have been exhausted.

    Args:
    in_data: The input data.
    out_data: The annotated outputs.
    instance_types_per_client: Number of intent types per client.
    instances_per_client: Instances per client.
    
    Returns:
    A dictionary of client ids mapped to the client dataset and
    the number of clients.
    """

    # get all unique intents
    unique_intents = np.unique(out_data[:, 1])
    np.random.shuffle(unique_intents)

    # group training instance by the intent_type
    intent_wise_data = {
        intent: np.where(out_data[:, 1] == intent)[0]
        for intent in unique_intents
    }
    sampling_intent_weight = {
        intent: len(intent_wise_data[intent])
        for intent in unique_intents
    }

    client_id = 0
    client_idxs = defaultdict(list)

    # Create new clients till the training set is exhausted
    while len(intent_wise_data):

        # If remaning intent types is less then use all remaining intents, else sample intents
        if len(intent_wise_data) < instance_types_per_client:
            client_intents = list(intent_wise_data.keys())
        else:
            prob_dist = list(sampling_intent_weight.values()) / (np.sum(
                list(sampling_intent_weight.values())))
            client_intents = np.random.choice(list(intent_wise_data.keys()),
                                              size=instance_types_per_client,
                                              replace=False,
                                              p=prob_dist)

        # Generate a random multinomial distribution over the selected subset of intents
        intent_client_distribution = np.random.randint(
            low=0, high=1000, size=len(client_intents)).astype(np.float)
        intent_client_distribution /= np.sum(intent_client_distribution)

        # Sample traning instances from the training set without replacement
        # using the random mutinomial distribution.
        for intent, dist in zip(client_intents, intent_client_distribution):

            required_instance_count = int(instances_per_client * dist)

            if required_instance_count > len(intent_wise_data[intent]):
                client_instances = np.arange(len(intent_wise_data[intent]))
            else:
                client_instances = np.random.choice(
                    len(intent_wise_data[intent]),
                    size=required_instance_count,
                    replace=False)

            # Add sampled indices to client index list
            client_idxs[client_id] += list(
                intent_wise_data[intent][client_instances])
            intent_wise_data[intent] = np.delete(intent_wise_data[intent],
                                                 client_instances)
            sampling_intent_weight[intent] -= len(client_instances)

            # delete intent when all instance of intent have been used
            if len(intent_wise_data[intent]) == 0:
                del intent_wise_data[intent]
                del sampling_intent_weight[intent]

        client_id += 1

    num_clients = client_id

    train_fed_data, valid_fed_data = create_tff_dataset(
        client_idxs, in_data, out_data, num_clients, p13n_train_ratio)

    return train_fed_data, valid_fed_data, num_clients
