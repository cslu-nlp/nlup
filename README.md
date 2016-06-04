`nlup` contains some base libraries I use in natural language processing projects. Some highlights:

* `confusion.py`: classifier evaluation objects
* `decorators.py`: clever decorators for various purposes
* `jsonable.py`: a mix-in which allows the state of most objects to be serialized to (and deserialized from) compressed JSON
* `perceptron.py`: perceptron-like classifiers (binary and multiclass), including some forms of structured prediction
* `reader.py`: classes and readers for tagged and dependency-parsed data
* `timer.py`: a `with`-block that logs wall clock time elapsed

All have been tested on CPython 3.4.1 and PyPy 3.2.5 (PyPy version 2.3.1). They will not work on Python 2 without modification.

Some projects using `nlup`:

* [Detector Morse](http://github.com/cslu-nlp/detectormorse): simple sentence boundary detection
* [Perceptronix Point Never](http://github.com/cslu-nlp/PerceptronixPointNever): simple part of speech tagging
* [Where's Yr Head At](http://github.com/cslu-nlp/WheresYrHeadAt): simple transition-based dependency parsing
