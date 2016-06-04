# Copyright (c) 2013-2016 Kyle Gorman
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
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""Confusion matrix classes."""


from math import sqrt
from functools import partial
from collections import defaultdict


NAN = float("nan")
INF = float("inf")


class ConfusionMixin(object):
  """Confusion methods shared by all three non-abstract classes."""

  def batch_update(self, truths, guesses):
    for (truth, guess) in zip(truths, guesses):
      self.update(truth, guess)

  @property
  def confint(self):
    """Computes 95% binomial confidence intervals.

    This method computes the 95% binomial confidence interval for sample
    accuracy using the Wilson method. The lower bound is particularly
    useful in system ranking.

    The return value is a (lower_bound, upper_bound) tuple.
    """
    if not len(self):
      return (0., 1.)
    n = len(self)
    phat = self.accuracy
    z = 1.9599639845400538273879   # -qnorm(.05 / 2)
    zsq = z * z
    a1 = 1. / (1. + zsq / n)
    a2 = phat + zsq / (2 * n)
    a3 = z * sqrt(phat * (1. - phat) / n + zsq / (4 * n * n))
    return (a1 * (a2 - a3), a1 * (a2 + a3))


class Accuracy(ConfusionMixin):
  """Accuracy for classification tasks."""

  def __init__(self, correct=0, incorrect=0):
    self.correct = correct
    self.incorrect = incorrect

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(Self))

  def __str__(self):
    return "{:.4f}".format(self.accuracy)

  def __len__(self):
    return self.correct + self.incorrect

  def outcome(self, is_hit):
    if is_hit:
      self.correct += 1
    else:
      self.incorrect += 1

  def update(self, truth, guess):
    self.outcome(truth == guess)

  def __add__(self, right):
    """Combines two Accuracy objects."""
    return Accuracy(self.correct + right.correct,
                    self.incorrect + right.incorrect)

  @property
  def accuracy(self):
    return self.correct / len(self)


class BinaryConfusion(ConfusionMixin):
  """Binary confusion matrix for classification tasks."""

  def __init__(self, hit=True, tp=0, fp=0, fn=0, tn=0):
    self.hit = hit
    self.tp = tp
    self.fp = fp
    self.fn = fn
    self.tn = tn

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(Self))

  def pprint(self):
    """
    >>> cm = BinaryConfusion()
    >>> cm.tp = 5809125
    >>> cm.tn = 2235458
    >>> cm.fp = cm.fn = 1
    >>> cm.pprint()
    Truth | Guess
    ---------------------------------------
          |       Hit         Miss
     Hit  | 5,809,125            1
     Miss |         1    2,235,458
    """
    print("""Truth | Guess
---------------------------------------
      |       Hit         Miss
 Hit  | {:>9,}    {:>9,}
 Miss | {:>9,}    {:>9,}""".format(self.tp, self.fn, self.fp, self.tn))

  @property
  def summary(self):
    return """Accuracy:\t{:.4f}
Precision:\t{:.4f}
Recall:\t\t{:.4f}
F1:\t\t{:.4f}""".format(self.accuracy, self.precision,
                        self.recall, self.F1)

  def __len__(self):
    return self.tp + self.fp + self.fn + self.tn

  def update(self, truth, guess):
    truth = bool(truth)
    guess = bool(guess)
    if truth:
      if guess:
        self.tp += 1
      else:
        self.fn += 1
    elif guess:
      self.fp += 1
    else:
      self.tn += 1

  def __add__(self, right):
    """Combines two BinaryConfusion objects."""
    if self.hit != right.hit:
      raise ValueError("Operands do not have matching 'hit' label.")
    return BinaryConfusion(hit=self.hit,
                           tp=self.tp + right.tp,
                           fp=self.fp + right.fp,
                           fn=self.fn + right.fn,
                           tn=self.tn + right.tn)

  @property
  def accuracy(self):
    try:
      return (self.tp + self.tn) / len(self)
    except ZeroDivisionError:
      return NAN

  @property
  def Kappa(self):
    """Cohen's Kappa for interannotator agreement."""
    # Probability the two sources say yes
    if len(self) == 0:
      return NAN
    Px = (self.tp + self.fp) / len(self)
    Py = (self.tp + self.fn) / len(self)
    # Probability of chance agreement.
    Pe = (Px * Py) + ((1. - Px) * (1. - Py))
    return (self.accuracy - Pe) / (1. - Pe)

  def Fscore(self, ratio=1.):
    """F-score, by default F_1 (equal weight to precision and recall)."""
    assert ratio > 0.
    r_square = ratio * ratio
    P = self.precision
    R = self.recall
    return ((1. + r_square) * P * R) / (r_square * P + R)

  @property
  def F1(self):
    return self.Fscore()

  def Sscore(self, ratio=1.):
    """S-score, by default S_1 (equal weight to specificity and sensitivity)."""
    assert ratio > 0.
    r_square = ratio * ratio
    Sp = self.specificity
    Se = self.sensitivity
    return ((1. + r_square) * Sp * Se) / (r_square * Sp + Se)

  @property
  def S1(self):
    return self.Sscore()

  @property
  def youden_J(self):
    """Youden's J statistic."""
    return self.sensitivity + self.specificity - 1.

  @property
  def MCC(self):
    try:
      N = len(self)
      S = (self.tp + self.fn) / N
      P = (self.tp + self.fp) / N
      PS = P * S
      denom = sqrt(PS * (1. - S) * (1. - P))
      return ((self.tp / N) - PS) / denom
    except ZeroDivisionError:
      return NAN

  # Precision.

  @property
  def precision(self):
    try:
      return self.tp / (self.tp + self.fp)
    except ZeroDivisionError:
      return INF

  @property
  def PPV(self):
    return self.precision

  # Recall.

  @property
  def recall(self):
    try:
      return self.tp / (self.tp + self.fn)
    except ZeroDivisionError:
      return INF

  @property
  def sensitivity(self):
    return self.recall

  @property
  def TPR(self):
    return self.recall

  # Specificity.

  @property
  def specificity(self):
    try:
      return self.tn / (self.tn + self.fp)
    except ZeroDivisionError:
      return INF

  @property
  def TNR(self):
    return self.specificity

  # Others, rarely used.

  @property
  def FPR(self):
    try:
      return self.fp / (self.fp + self.tn)
    except ZeroDivisionError:
      return INF

  @property
  def NPV(self):
    try:
      return self.tn / (self.tn + self.fn)
    except ZeroDivisionError:
      return INF

  @property
  def FDR(self):
    return 1. - self.PPV


class Confusion(ConfusionMixin):
  """Generic multiclass confusion matrix."""

  def __init__(self):
    self.matrix = defaultdict(partial(defaultdict, int))
    self.correct = 0
    self.incorrect = 0

  def __len__(self):
    return self.correct + self.incorrect

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(Self))

  def pprint(self):
    print("Confusion matrix:")
    for (truth, guess_count) in self.matrix.items():
      print("{}:".format(truth))
      for (guess, count) in guess_count.items():
        print("\t{}: {:,}".format(guess, count))

  def update(self, truth, guess, k=1):
    self.matrix[truth][guess] += k
    if truth == guess:
      self.correct += 1
    else:
      self.incorrect += 1

  def __add__(self, right):
    """Combines two Confusion objects."""
    # Constructs new Confusion instance.
    retval = Confusion()
    retval.matrix = self.matrix.copy()
    for (truth, guess_count) in right.matrix.items():
      truth_ptr = retval.matrix[truth]
      for (guess, count) in guess_count.items():
        truth_ptr[guess] += count
      retval.correct = self.correct + right.correct
      retval.incorrect = self.incorrect + right.incorrect
      return retval

  @property
  def accuracy(self):
    return self.correct / (self.correct + self.incorrect)

  @property    
  def Kappa(self):
    """Cohen's Kappa for interannotator agreement."""
    Pe = 0.
    Z = sum(sum(gf.values()) for gf in self.matrix.values())
    for (truth, guess_count) in self.matrix.items():
       Pe_i = sum(guess_count.values()) / Z
       Pe += Pe_i * Pe_i
    return (self.accuracy - Pe) / (1. - Pe)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
