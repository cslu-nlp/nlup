`nlup` contains some base libraries I use in natural language understanding projects. Some highlights:

* `perceptron.py`: perceptron-like classifiers (binary and multiclass using the one-versus-all strategy), including some forms of structured prediction
* `confusion.py`: classifier evaluation objects
* `decorators.py`: clever decorators for various purposes
* `jsonable.py`: a mix-in which allows the state of most objects to be serialized to (and deserialized from) compressed JSON

All have been tested on CPython 3.4.1 and PyPy 3.2.5 (PyPy version 2.3.1). They will not work on 
