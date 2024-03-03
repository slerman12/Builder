# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import numpy as np


"""
Examples:
"""


class Accuracy:
    # An "exp" (experience) is a key-value structure of batch data that follows an action
    def add(self, exp):
        return exp.label == exp.action  # Gets appended to an epoch list

    # At the end of an epoch, a metric is tabulated
    def tabulate(self, epoch):
        return epoch  # By default, lists/arrays get (1) concatenated, and (2) mean-averaged


class MSE:
    def add(self, exp):
        return (exp.label - exp.action) ** 2  # Gets appended to an epoch list

    def tabulate(self, epoch):
        return epoch  # By default, lists/arrays get (1) concatenated, and (2) mean-averaged


class Reward:  # By default, the Environment tabulate_metric already does this if "reward" & no reward metric passed in
    def add(self, exp):
        return np.array(exp.reward).reshape(-1)

    def tabulate(self, episode):  # At the end of an episode, reward is summed
        return np.concatenate(episode).sum()


class Precision:
    def add(self, exp):
        classes = np.unique(exp.action)

        # True positives are the number of correct predictions for a class
        true_positives = {c: sum((exp.action == exp.label) & (exp.action == c)) for c in classes}

        # For Precision, total is the number of predictions for the class
        total = {c: sum(exp.action == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average precision (only works for binary classification, e.g., dataset.subset='[0,1]')
        c = next(iter(epoch[0][0].keys()))  # Use first class, e.g., 0
        return sum([true_positives[c] for true_positives, _ in epoch if c in true_positives]) \
            / sum([total[c] for true_positives, total in epoch if c in true_positives])

        # Macro-average precision  TODO For macro-average, get "num_classes"
        #                               and divide sum-of-precision-for-each-class by "num_classes"
        # return sum([true_positives[c] for true_positives, _ in epoch for c in true_positives]) \
        #     / sum([total[c] for _, total in epoch for c in total])


class Recall:
    def add(self, exp):
        classes = np.unique(exp.label)

        # True positives are the number of correct predictions for a class
        true_positives = {c: sum((exp.action == exp.label) & (exp.action == c)) for c in classes}

        # For Recall, total is the number of labels for the class
        total = {c: sum(exp.label == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average recall (only works for binary classification, e.g., dataset.subset='[0,1]')
        c = next(iter(epoch[0][0].keys()))  # Use first class, e.g., 0
        return sum([true_positives[c] for true_positives, _ in epoch if c in true_positives]) \
            / sum([total[c] for true_positives, total in epoch if c in true_positives])

        # Macro-average recall  TODO For macro-average, get "num_classes"
        #                               and divide sum-of-precision-for-each-class by "num_classes"
        # return sum([true_positives[c] for true_positives, _ in epoch for c in true_positives]) \
        #     / sum([total[c] for _, total in epoch for c in total])


"""
    Call via command-line as follows: 
    
        ML metric.precision=World.Metrics.Precision metric.recall=World.Metrics.Recall

    -------------------------------------------------
    
    Supports math string expressions as well. Example:
    
    F1-Score:
    
        metric.F1='2*precision*recall/(precision+recall)'
    
    Try calling via pure command line:
    
        ML metric.precision=World.Metrics.Precision metric.recall=World.Metrics.Recall metric.F1='2*precision*recall/(precision+recall)'
"""
