#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn.cross_validation import train_test_split  # to split the dataset for training and testing

        train, test = train_test_split(self.marvin_initial_dataset, test_size=0.3)  # in this our main data is split into train and test

        # the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
        print(train.shape)
        print(test.shape)

        train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # taking the training data features
        train_Y = train.Species  # output of our training data

        test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # taking test data features
        test_Y = test.Species  # output value of test data

        self.marvin_dataset = {
            "train_X": train_X,
            "train_Y": train_Y,
            "test_X": test_X,
            "test_Y": test_Y,
        }

