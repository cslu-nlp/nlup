"""
Objects representing tagged or dependency-parsed sentences
"""

from nltk import str2tuple, tuple2str


class TaggedSentence(object):
    """
    Part-of-speech tagged data in `token/tag` format
    """

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
        return " ".join(tuple2str(tt) for tt in \
                        zip(self.tokens, self.tags))


class DependencyParsedSentence(object):
    """
    Dependency parsed data in `token\ttag\thead\tlabel` format
    """

    def __init__(self, tokens, tags, heads, labels=None):
        self.tokens = tokens
        self.tags = tags
        self.heads = heads
        self.labels = labels
        assert len(self.tokens) == \
            len(self.tags)   == \
            len(self.heads)  == \
            len(self.labels)

    @classmethod
    def from_str(cls, string):
        bits = zip(*(line.split() for line in string.splitlines()))
        (tokens, tags, heads, labels) = bits
        return cls(tokens, tags, heads, labels)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return "{}(tokens={!r}, tags={!r}, heads={!r}, labels={!r}".format(self.__class__.__name__, self.tokens, self.tags, self.heads, self.labels)

    def __str__(self):
        lines = zip(self.tokens, self.tags, self.heads, self.labels)
        return "\n".join("\t".join(line) for line in lines)


class ConstituencyParsedSentence(object):

    def __init__(self):
        raise NotImplementedError
