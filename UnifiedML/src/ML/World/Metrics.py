import numpy as np


class Accuracy:  # TODO Debug discrete = false  (Need to argmax)
    # An experience is a set of batch data that follows an action
    def add(self, exp):
        return exp.label == exp.action.squeeze(-1)  # Gets appended to an epoch list

    # At the end of an epoch, a metric is tabulated
    def tabulate(self, epoch):
        return epoch  # Lists/arrays get concatenated and mean-averaged by default


class MSE:
    def add(self, exp):
        return (exp.label - exp.action.squeeze(1)) ** 2  # Gets appended to an epoch list

    def tabulate(self, epoch):
        return epoch  # Lists/arrays get concatenated and mean-averaged by default


class Reward:
    def add(self, exp):
        if 'reward' in exp:
            # Note: Taking mean batch-wise assumes each batch same size
            return exp.reward.mean() if hasattr(exp.reward, 'mean') else exp.reward

    def tabulate(self, episode):  # At the end of an episode, a metric is tabulated
        episode = [reward for reward in episode if reward is not None]

        if episode:
            return sum(episode)


class Precision:
    def add(self, exp):
        classes = np.unique(exp.action)

        true_positives = {c: exp.action == exp.label == c for c in classes}
        total = {c: sum(exp.action == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average precision
        return sum([value[key][0] for value in epoch for key in value]) \
            / sum([value[key][1] for value in epoch for key in value])


class Recall:
    def add(self, exp):
        classes = np.unique(exp.label)

        true_positives = {c: exp.action == exp.label == c for c in classes}
        total = {c: sum(exp.label == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average precision
        return sum([value[key][0] for value in epoch for key in value]) \
            / sum([value[key][1] for value in epoch for key in value])


# metric.F1=2*precision*recall/(precision+recall)
