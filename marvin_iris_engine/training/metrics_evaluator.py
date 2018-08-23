#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn import metrics  # for checking the model accuracy

        prediction = self.marvin_model.predict(self.marvin_dataset["test_X"])  # now we pass the testing data to the trained algorithm
        metrics = metrics.accuracy_score(prediction, self.marvin_dataset["test_Y"])

        self.marvin_metrics = {
            "accuracy": float(metrics)
        }

        print('The accuracy of the SVM is:', metrics)  # now we check the accuracy of the algorithm.

