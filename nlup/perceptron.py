# Copyright (C) 2014-2015 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# PLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
perceptron: perceptron-like classifers, including:

* `BinaryPerceptron`: binary perceptron classifier
* `Perceptron`: multiclass perceptron classifier
* `SequencePerceptron`: multiclass perceptron for sequence tagging
* `BinaryAveragedPerceptron`: binary averaged perceptron classifier
* `AveragedPerceptron`: multiclass averaged perceptron
* `SequenceAveragedPerceptron`: multiclass averaged perceptron for
   sequence tagging
"""


import logging

from random import Random
from functools import partial
from operator import itemgetter
from collections import defaultdict, namedtuple

from .timer import Timer
from .jsonable import JSONable
from .confusion import Accuracy
from .decorators import reversify, tupleify


INF = float("inf")
ORDER = 0
EPOCHS = 1


class Classifier(JSONable):

    """
    Mixin for shared classifier methods
    """

    def fit(self, Y, Phi, epochs=EPOCHS, alpha=1):
        data = list(zip(Y, Phi))  # which is a copy
        logging.info("Starting {} epoch(s) of training.".format(epochs))
        for epoch in range(1, 1 + epochs):
            logging.info("Epoch {:>2}.".format(epoch))
            accuracy = Accuracy()
            self.random.shuffle(data)
            with Timer():
                for (y, phi) in data:
                    yhat = self.fit_one(y, phi, alpha)
                    accuracy.update(y, yhat)
                logging.info("Accuracy: {!s}".format(accuracy))
        self.finalize()

    def finalize(self):
        pass


class BinaryPerceptron(Classifier):

    """
    Binary perceptron classifier
    """

    def __init__(self, seed=None):
        self.random = Random(seed)
        self.weights = defaultdict(int)

    def score(self, phi):
        """
        Get score for a `hit` according to the feature vector `phi`
        """
        return sum(self.weights[feature] for feature in phi)

    def predict(self, phi):
        """
        Predict binary decision for the feature vector `phi`
        """
        return self.score(phi) >= 0

    def fit_one(self, y, phi, alpha=1):
        yhat = self.predict(phi)
        if y != yhat:
            self.update(y, phi, alpha)
        return yhat

    def update(self, y, phi, alpha=1):
        """
        Given feature vector `y`, reward correct observation `y` with
        the update `alpha`
        """
        assert y in {True, False}
        assert 0. < alpha <= 1.
        if y is False:
            alpha *= -1
        for phi_i in phi:
            self.weights[phi] += alpha


class Perceptron(Classifier):

    """
    The multiclass perceptron with sparse binary feature vectors:

    Each class (i.e., label, outcome) is represented as a hashable item,
    such as a string. Features are represented as hashable objects
    (preferably strings, as Python dictionaries have been aggressively
    optimized for this case). Presence of a feature indicates that that
    feature is "firing" and absence indicates that that it is not firing.
    This class is primarily to be used as an abstract base class; in most
    cases, the regularization and stability afforded by the averaged
    perceptron (`AveragedPerceptron`) will be worth it.

    The perceptron was first proposed in the following paper:

    F. Rosenblatt. 1958. The perceptron: A probabilistic model for
    information storage and organization in the brain. Psychological
    Review 65(6): 386-408.
    """

    # constructor

    def __init__(self, classes=(), seed=None):
        self.classes = tuple(classes)
        self.random = Random(seed)
        self.weights = defaultdict(partial(defaultdict, int))

    def register_classes(self, classes):
        """
        Register class labels in classifier instance
        """
        self.classes = tuple(classes)

    def score(self, y, phi):
        """
        Get score for one class (`y`) according to the feature vector
        `phi`
        """
        assert self.classes
        return sum(self.weights[phi_i][y] for phi_i in phi)

    def scores(self, phi):
        """
        Get scores for all classes according to the feature vector `phi`
        """
        assert self.classes
        scores = dict.fromkeys(self.classes, 0)
        for phi_i in phi:
            for (cls, weight) in self.weights[phi_i].items():
                scores[cls] += weight
        return scores

    def predict(self, phi):
        """
        Predict most likely class for the feature vector `phi`
        """
        scores = self.scores(phi)
        (argmax_score, _) = max(scores.items(), key=itemgetter(1))
        return argmax_score

    def fit_one(self, y, phi, alpha=1):
        yhat = self.predict(phi)
        if y != yhat:
            self.update(y, yhat, phi, alpha)
        return yhat

    def update(self, y, yhat, phi, alpha=1):
        """
        Given feature vector `x`, reward correct observation `y` and
        punish incorrect hypothesis `yhat` with the update `alpha`
        """
        for phi_i in phi:
            ptr = self.weights[phi_i]
            ptr[y] += alpha
            ptr[yhat] -= alpha


TrellisCell = namedtuple("TrellisCell", ["score", "pointer"])


class SequencePerceptron(Perceptron):

    """
    Perceptron with Viterbi-decoding powers
    """

    def __init__(self, efeats_fnc, tfeats_fnc, order=ORDER, **kwargs):
        super(SequencePerceptron, self).__init__(**kwargs)
        self.efeats_fnc = efeats_fnc
        self.tfeats_fnc = tfeats_fnc
        self.order = order

    def predict(self, xx):
        """
        Tag a sequence using a greedy approximation of the Viterbi 
        algorithm, in which each sequence is tagged using transition
        features based on earlier hypotheses. The time complexity of this 
        operation is O(nt) where n is sequence length and t is the 
        cardinality of the tagset. 
        """
        (yyhat, _) = self._greedy_predict(xx)
        return yyhat

    def predict_with_transitions(self, xx):
        """
        Same as above, but hacked to give you the features back
        """
        return self._greedy_predict(xx)

    def _greedy_predict(self, xx):
        """
        Sequence classification with a greedy approximation of a Markov
        model, also returning feature vectors `phiphi`
        """
        yyhat = []
        phiphi = []
        for phi in self.efeats_fnc(xx):
            phi = phi + self.tfeats_fnc(yyhat[-self.order:])
            (yhat, _) = max(self.scores(phi).items(), key=itemgetter(1))
            yyhat.append(yhat)
            phiphi.append(phi)
        return (tuple(yyhat), tuple(phiphi))

    def fit_one(self, yy, xx, alpha=1):
        # decode to get predicted sequence
        (yyhat, phiphi) = self.predict_with_transitions(xx)
        for (y, yhat, phi) in zip(yy, yyhat, phiphi):
            if y != yhat:
                self.update(y, yhat, phi, alpha)
        return yyhat

    def fit(self, YY, XX, epochs=EPOCHS, alpha=1):
        data = list(zip(YY, XX))
        logging.info("Starting {} epoch(s) of training.".format(epochs))
        for epoch in range(1, 1 + epochs):
            logging.info("Epoch {:>2}.".format(epoch))
            accuracy = Accuracy()
            self.random.shuffle(data)
            with Timer():
                for (yy, xx) in data:
                    yyhat = self.fit_one(yy, xx, alpha)
                    accuracy.batch_update(yy, yyhat)
                logging.info("Accuracy: {!s}".format(accuracy))
        self.finalize()


class LazyWeight(object):

    """
    Helper class for `AveragedPerceptron`:

    Instances of this class are essentially triplets of values which
    represent a weight of a single feature in an averaged perceptron.
    This representation permits "averaging" to be done implicitly, and
    allows us to take advantage of sparsity in the feature space.
    First, as the name suggests, the `summed_weight` variable is lazily
    evaluated (i.e., computed only when needed). This summed weight is the
    one used in actual inference: we need not average explicitly. Lazy
    evaluation requires us to store two other numbers. First, we store the
    current weight, and the last time this weight was updated. When we
    need the real value of the summed weight (for inference), we "freshen"
    the summed weight by adding to it the product of the real weight and
    the time elapsed.

    # initialize
    >>> t = 0
    >>> lw = LazyWeight(t=t)
    >>> t += 1
    >>> lw.update(t, 1)
    >>> t += 1
    >>> lw.get()
    1

    # some time passes...
    >>> t += 1
    >>> lw.get()
    1

    # weight is now changed
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.get()
    -1
    """

    def __init__(self, default_factory=int, t=0):
        self.timestamp = t
        self.weight = default_factory()
        self.summed_weight = default_factory()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def get(self):
        """
        Return current weight
        """
        return self.weight

    def _freshen(self, t):
        """
        Apply queued updates, and update the timestamp
        """
        self.summed_weight += (t - self.timestamp) * self.weight
        self.timestamp = t

    def update(self, value, t):
        """
        Bring sum of weights up to date, then add `value` to the weight
        """
        self._freshen(t)
        self.weight += value

    def average(self, t):
        """
        Set `self.weight` to the summed value, for final inference
        """
        self._freshen(t)
        self.weight = self.summed_weight / t


class BinaryAveragedPerceptron(BinaryPerceptron):

    def __init__(self, seed=None):
        self.random = Random(seed)
        self.weights = defaultdict(LazyWeight)
        self.time = 0

    def predict(self, phi):
        """
        Predict most likely class for the feature vector `phi`
        """
        score = sum(self.weights[feature].get() for feature in phi)
        return score >= 0

    def fit_one(self, y, phi, alpha=1):
        retval = super(BinaryAveragedPerceptron, self).fit_one(y, phi,
                                                               alpha)
        self.time += 1
        return retval

    def update(self, y, phi, alpha=1):
        """
        Given feature vector `phi`, reward correct observation `y` and
        punish incorrect hypothesis `yhat` with the update `alpha`, 
        assuming that `y != yhat`.
        """
        assert y in {True, False}
        assert 0. < alpha <= 1.
        if y is False:
            alpha *= -1
        for phi_i in phi:
            self.weights[phi_i].update(alpha, self.time)

    def finalize(self):
        """
        Prepare for inference by removing zero-valued weights and applying
        averaging

        TODO(kbg): also remove zero-valued weights?
        """
        for (feature, weight) in self.weights.items():
            weight.average(self.time)


class AveragedPerceptron(Perceptron):

    """
    The multiclass perceptron with sparse binary feature vectors, with
    averaging for stability and regularization.

    Averaging was originally proposed in the following paper:

    Y. Freund and R.E. Schapire. 1999. Large margin classification using
    the perceptron algorithm. Machine Learning 37(3): 227-296.
    """

    def __init__(self, classes=(), seed=None):
        self.classes = tuple(classes)
        self.random = Random(seed)
        self.weights = defaultdict(partial(defaultdict, LazyWeight))
        self.time = 0

    def score(self, y, phi):
        """
        Get score for one class (`y`) according to the feature vector 
        `phi`
        """
        return sum(self.weights[phi_i][y].get() for phi_i in phi)

    def scores(self, phi):
        """
        Get scores for all classes according to the feature vector `phi`
        """
        scores = dict.fromkeys(self.classes, 0)
        for phi_i in phi:
            for (cls, weight) in self.weights[phi_i].items():
                scores[cls] += weight.get()
        return scores

    def fit_one(self, y, phi, alpha=1):
        retval = super(AveragedPerceptron, self).fit_one(y, phi, alpha)
        self.time += 1
        return retval

    def update(self, y, yhat, phi, alpha=1):
        """
        Given feature vector `phi`, reward correct observation `y` and
        punish incorrect hypothesis `yhat` with the update `alpha`
        """
        for phi_i in phi:
            ptr = self.weights[phi_i]
            ptr[y].update(+alpha, self.time)
            ptr[yhat].update(-alpha, self.time)

    def finalize(self):
        """
        Prepare for inference by removing zero-valued weights and applying
        averaging

        TODO(kbg): also remove zero-valued weights?
        """
        for (phi_i, clsweights) in self.weights.items():
            for (cls, weight) in clsweights.items():
                weight.average(self.time)


class SequenceAveragedPerceptron(AveragedPerceptron, SequencePerceptron):

    def __init__(self, efeats_fnc, tfeats_fnc, order=ORDER, **kwargs):
        super(SequenceAveragedPerceptron, self).__init__(**kwargs)
        self.efeats_fnc = efeats_fnc
        self.tfeats_fnc = tfeats_fnc
        self.order = order
