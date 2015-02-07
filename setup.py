from setuptools import setup


setup(name="nlup",
      version="0.3",
      description="Core libraries for natural language processing, with perceptron classifiers",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      url="http://github.com/cslu-nlp/nlup/",
      install_requires=["jsonpickle >= 0.8.0"],
      packages=["nlup"])
