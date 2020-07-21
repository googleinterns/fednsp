"""This file defines tests for the functions in utils.py.

The test cases and the required tests for evaluating metrics
and generating masks are tested in this file.
"""

import math
import pytest
import numpy as np
from util import compute_f1, compute_semantic_acc, create_masks

SEMANTIC_ACCURACY_TESTS = [
    # case where all intents and slots predictions match with GT
    (np.array([[5, 6, 9, 7, 3], [4, 3, 4, 8, 9]]), np.array([2, 1]),
     np.array([[5, 6, 9, 7, 3], [4, 3, 4, 8, 9]]), np.array([2, 1]), 100.0),

    # case where all intents are predicted correctly but some are predicted incorrectly
    (np.array([[5, 6, 9, 7, 3], [4, 3, 4, 8, 9]]), np.array([2, 1]),
     np.array([[5, 6, 8, 7, 3], [4, 2, 4, 8, 9]]), np.array([2, 1]), 0.0),

    # case where all slots are predicted correctly but intents are predicted incorrectly
    (np.array([[5, 6, 9, 7, 11], [4, 3, 11, 8, 12]]), np.array([2, 1]),
     np.array([[5, 6, 9, 7, 11], [4, 3, 11, 8, 12]]), np.array([1, 2]), 0.0),

    # case where some slots and some intents are predicted incorrectly
    (np.array([[5, 6, 9, 10, 11], [4, 3, 11, 8, 12]]), np.array([2, 1]),
     np.array([[5, 6, 9, 10, 11], [3, 3, 11, 8, 12]]), np.array([1, 1]), 0.0),

    # case where some slots and some intents are predicted incorrectly
    (np.array([[5, 6, 9, 10, 11], [4, 3, 11, 8, 12]]), np.array([2, 1]),
     np.array([[5, 6, 9, 10, 11], [3, 3, 11, 8, 12]]), np.array([2, 2]), 50.0),

    # case to test if padding tokes are ignored
    (np.array([[5, 6, 0, 0, 0], [4, 3, 11, 0, 0]]), np.array([2, 1]),
     np.array([[5, 6, 9, 10, 11], [4, 3, 11, 8, 12]]), np.array([2,
                                                                 1]), 100.0),
]

F1_SCORE_TESTS = [
    # simple test case where all the slots match
    (['B-toloc.city_name',
      'I-toloc.city_name'], ['B-toloc.city_name',
                             'I-toloc.city_name'], 100.0, 100.0, 100.0),

    # test case where all the slots match with 'O' in between chunks
    ([
        'B-toloc.city_name', 'O', 'O', 'B-flight_time', 'I-flight_time', 'O',
        'B-city_name'
    ], [
        'B-toloc.city_name', 'O', 'O', 'B-flight_time', 'I-flight_time', 'O',
        'B-city_name'
    ], 100.0, 100.0, 100.0),

    # another test case where all the slots match
    ([
        'O', 'B-depart_date.month_name', 'B-depart_date.day_number', 'O',
        'B-cost_relative', 'I-cost_relative', 'B-depart_time.time_relative'
    ], [
        'O', 'B-depart_date.month_name', 'B-depart_date.day_number', 'O',
        'B-cost_relative', 'I-cost_relative', 'B-depart_time.time_relative'
    ], 100.0, 100.0, 100.0),

    # test case where there is an alignment mismatch
    (['O', 'O', 'O', 'B-city_name', 'O'], ['O', 'O', 'B-city_name', 'O',
                                           'O'], 0.0, 0.0, 0.0),

    # test case where an O is misclassified (false-positive)
    (['O', 'O', 'B-aircraft_code',
      'O'], ['O', 'O', 'B-aircraft_code',
             'B-aircraft_code'], 50.0, 100.0, 200.0 / 3),

    # # test case where the wrong I-tag is predicted after a correct B-tag
    # (['B-toloc.city_name', 'O', 'O', 'B-flight_time', 'I-flight_time', 'O',  'B-city_name'],
    #  ['B-toloc.city_name', 'O', 'O', 'B-flight_time', 'I-flight_mod', 'O',  'B-city_name'],
    #  200.0/3, 200.0/3, 200.0/3),

    # test case where tags are  misclassified
    ([
        'B-fromloc.city_name', 'O', 'B-toloc.city_name',
        'B-depart_date.today_relative', 'O', 'B-meal'
    ], [
        'B-fromloc.city_name', 'O', 'B-fromloc.city_name',
        'B-depart_date.date_relative', 'O', 'B-meal'
    ], 50.0, 50.0, 50.0),

    # test case where an I-tag are predicted instead of a B-tag
    # (['O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'I-toloc.city_name'],
    #  ['O', 'B-fromloc.city_name', 'O', 'I-toloc.city_name', 'I-toloc.city_name'],
    #  50.0, 50.0, 50.0),

    # test case where all tags are predicted as O
    (['B-fromloc.city_name', 'O', 'B-toloc.city_name',
      'B-meal'], ['O', 'O', 'O', 'O'], 0.0, 0.0, 0.0),

    # test case where all ground truth tags are 'O'
    (['O', 'O', 'O',
      'O'], ['B-fromloc.city_name', 'O', 'B-toloc.city_name',
             'B-meal'], 0.0, 0.0, 0.0),
]

CREATE_MASKS_TESTS = [
    # case where both the inputs have padding tokes
    (np.array([[5, 6, 9, 0], [4, 3, 0,
                              0]]), np.array([[6, 9, 0, 0], [3, 0, 0, 0]]),
     np.array([[[[0, 0, 0, 1]]], [[[0, 0, 1, 1]]]]),
     np.array([[[1], [1], [1], [0]], [[1], [1], [0], [0]]]),
     np.maximum([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[[[0, 0, 1, 1]]], [[[0, 1, 1, 1]]]])),

    # case where both the inputs don't have padding tokes
    (np.array([[5, 6, 9], [4, 3, 1]]), np.array([[6, 9, 3], [3, 1, 2]]),
     np.array([[[[0, 0, 0]]],
               [[[0, 0, 0]]]]), np.array([[[1], [1], [1]], [[1], [1], [1]]]),
     np.maximum([[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                [[[[0, 0, 0]]], [[[0, 0, 0]]]])),

    # case where one of them has padding tokens and the other does not
    (np.array([[5, 3, 0], [4, 3, 1]]), np.array([[3, 0, 0], [3, 1, 2]]),
     np.array([[[[0, 0, 1]]],
               [[[0, 0, 0]]]]), np.array([[[1], [1], [0]], [[1], [1], [1]]]),
     np.maximum([[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                [[[[0, 1, 1]]], [[[0, 0, 0]]]])),
]


@pytest.mark.parametrize(
    "gt_slots, pred_slots, expected_precision, expected_recall, expected_f1_score",
    F1_SCORE_TESTS)
def test_f1_score(gt_slots, pred_slots, expected_precision, expected_recall,
                  expected_f1_score):
    """Ensures that the precision, recall and the f1 score computed by
    the compute_f1() function are as expected.
    """

    computed_f1_score, computed_precision, computed_recall = compute_f1(
        [gt_slots], [pred_slots])

    assert math.isclose(expected_precision, computed_precision)
    assert math.isclose(expected_recall, computed_recall)
    assert math.isclose(expected_f1_score, computed_f1_score)


@pytest.mark.parametrize(
    "gt_slots, gt_intents, pred_slots, pred_intents, expected_accuracy",
    SEMANTIC_ACCURACY_TESTS)
def test_sematic_acc(gt_slots, gt_intents, pred_slots, pred_intents,
                     expected_accuracy):
    """Ensures that the semantic accuracy computed by
    the compute_semantic_acc() function is as expected.
    """

    computed_accuracy = compute_semantic_acc(gt_slots, gt_intents, pred_slots,
                                             pred_intents)

    assert math.isclose(expected_accuracy, computed_accuracy)


@pytest.mark.parametrize(
    "inputs, targets, expected_padding_mask, expected_intent_mask, expected_combined_mask",
    CREATE_MASKS_TESTS)
def test_mask_generation(inputs, targets, expected_padding_mask,
                         expected_intent_mask, expected_combined_mask):
    """Ensures that the masks generated by the create_masks() function are
    correct.
    """
    generated_padding_mask, generated_combined_mask, generated_intent_mask = create_masks(
        inputs, targets)

    assert np.alltrue(generated_padding_mask == expected_padding_mask)
    assert np.alltrue(generated_intent_mask == expected_intent_mask)
    assert np.alltrue(generated_combined_mask == expected_combined_mask)
