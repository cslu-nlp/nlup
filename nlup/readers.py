# Copyright (c) 2015-2016 Kyle Gorman
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


from nltk import str2tuple
from nltk import tuple2str


def untagged_corpus(filename):
  """Reads and yield token lists a `filename` string."""
  with open(filename, "r") as source:
    for line in source:
      line = line.strip()
      if not line:
        continue
      yield line.split()


class TaggedSentence(object):
  """Part-of-speech tagged data in `token/tag` format."""

  def __init__(self, tokens, tags):
    self.tokens = tokens
    self.tags = tags
    assert len(self.tokens) == len(self.tags)

  @classmethod
  def from_str(cls, string):
    (tokens, tags) = zip(*(str2tuple(tt) for tt in string.split()))
    return cls(tokens, tags)

  def __len__(self):
    return len(self.tokens)

  def __repr__(self):
    return "{}(tokens={!r}, tags={!r})".format(self.__class__.__name__,
                                               self.tokens, self.tags)

  def __str__(self):
    return " ".join(tuple2str(tt) for tt in zip(self.tokens, self.tags))

  def __iter__(self):
    return iter(zip(self.tokens, self.tags))


def tagged_corpus(filename):
  """Reads and yields TaggedSentence objects from a file."""
  with open(filename, "r") as source:
    for line in source:
      line = line.strip()
      if not line:
        continue
      yield TaggedSentence.from_str(line)


# TODO(kbg) implement these.


class ChunkedSentence(object):
  """NP-chunking data in token\ttag\tchunk format."""

  def __init__(self, tokens, tags, chunks):
    self.tokens = tokens
    self.tags = tags
    self.chunks = chunks
    assert len(self.tokens) == len(self.tags) == len(self.chunks)

  def __len__(self):
    return len(self.tokens)

  def __repr__(self):
    return "{}(tokens={!r}, tags={!r}, chunks={!r})".format(
        self.__class__.__name__, self.tokens, self.tags, self.chunks)

  def __str__(self):
    return "\n".join("\t".join(pieces) for pieces in self)

  def __iter__(self):
    return iter(zip(self.tokens, self.tags, self.chunks))


def chunked_corpus(filename):
  """Reads and yields ChunkedSentence objects from a file."""
  with open(filename, "r") as source:
    data = []
    for line in source:
      line = line.rstrip()
      if not line:
        yield ChunkedSentence(*zip(*data))
      else:
        (token, tags, chunk) = line.split()
        data.append((token, tags, chunk))
    yield ChunkedSentence(*zip(*data))


class DependencyParsedSentence(object):
  """Dependency parse data in token\ttag\thead\tlabel format."""

  def __init__(self, tokens, tags, heads, labels=None):
    self.tokens = tokens
    self.tags = tags
    self.heads = heads
    self.labels = labels
    assert (len(self.tokens) == len(self.tags) == len(self.heads) ==
            len(self.labels))

  @classmethod
  def from_str(cls, string):
    bits = zip(*(line.split() for line in string.splitlines()))
    (tokens, tags, heads, labels) = bits
    heads = tuple(int(i) for i in heads)
    return cls(tokens, tags, heads, labels)

  def __len__(self):
    return len(self.tokens)

  def __repr__(self):
    return "{}(tokens={!r}, tags={!r}, heads={!r}, labels={!r}".format(
        self.__class__.__name__, self.tokens, self.tags, self.heads, self.labels)

  def __str__(self):
    heads = tuple(str(i) for i in self.heads)
    lines = zip(self.tokens, self.tags, heads, self.labels)
    return "\n".join("\t".join(line) for line in lines)

  def latex_str(self, labels=False):
    """Prints as a string for a LaTeX table."""
    edges = ""
    for (i, (head, label)) in enumerate(zip(self.heads, self.labels)):
      edges += "\n    \\depedge{{{}}}{{{}}}{{{}}}".format(head, i + 1, label)
    return """\\begin{{dependency}}[theme=default]
    \\begin{{deptext}}[column sep=1em, row sep=1em]
    {} \\& ROOT \\\\
    \\end{{deptext}}{}
\\end{{dependency}}""".format(" \\& ".join(self.tokens), edges)

  def __iter__(self):
    return iter(zip(self.tokens, self.tags, self.heads, self.labels))


def depparsed_corpus(filename):
  """Reads and yields DependencyParseSentence objects from a file."""
  with open(filename, "r") as source:
    lines = []
    for line in source:
      line = line.strip()
      if not line:
        yield DependencyParsedSentence.from_str("\n".join(lines))
        lines = []
        continue
      lines.append(line)
    if lines:
      yield DependencyParsedSentence.from_str("\n".join(lines))


# TODO(kbg) implement these.


class ConstituencyParsedSentence(object):

  def __init__(self):
    raise NotImplementedError


def conparsed_reader(filename):
  raise NotImplementedError
