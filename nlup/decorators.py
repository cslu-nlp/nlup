# Copyright (C) 2014-2016 Kyle Gorman
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


"""Cute decorators."""


from functools import wraps
from .timer import Timer


def listify(gen):
  """Converts a generator into a function which returns a list."""
  @wraps(gen)
  def patched(*args, **kwargs):
    return list(gen(*args, **kwargs))
  return patched


def reversify(fnc):
  """Converts a function or generator which returns an iterable to one which
  returns a reversed list.

  This is accomplished by casting the iterable to a list (evaluating it in
  place, if necessary) and then reversing the result.
  """
  @wraps(fnc)
  def patched(*args, **kwargs):
    retval = list(fnc(*args, **kwargs))
    retval.reverse()
    return retval
  return patched


def tupleify(gen):
  """Converts a generator into a function which returns a tuple."""
  @wraps(gen)
  def patched(*args, **kwargs):
    return tuple(gen(*args, **kwargs))
  return patched


def setify(gen):
  """Converts a generator into a function which returns a set."""
  @wraps(gen)
  def patched(*args, **kwargs):
    return set(gen(*args, **kwargs))
  return patched


def frozensetify(gen):
  """Converts a generator into a function which returns a frozenset."""
  @wraps(gen)
  def patched(*args, **kwargs):
    return frozenset(gen(*args, **kwargs))
  return patched


def meanify(gen):
  """Converts a generator of numbers to one which returns the mean thereof.

  The algorithm used here is an online, numerically stable method described in
  AoCP (2.4.2.2).
  """
  @wraps(gen)
  def patched(*args, **kwargs):
    avg = 0
    for (i, val) in enumerate(gen(*args, **kwargs), 1):
      avg += (val - avg) / i
    return avg
  return patched


def timeify(fnc):
  """Converts a function to one which times its own execution using a logger."""
  @wraps(fnc)
  def patched(*args, **kwargs):
    with Timer():
      retval = fnc(*args, **kwargs)
    return retval
  return patched
