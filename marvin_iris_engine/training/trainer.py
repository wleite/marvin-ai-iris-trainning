#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn import svm  # for Support Vector Machine (SVM) Algorithm

        model = svm.SVC()  # select the algorithm
        model.fit(self.marvin_dataset["train_X"], self.marvin_dataset["train_Y"])  # we train the algorithm with the training data and the training output

        self.marvin_model = model

