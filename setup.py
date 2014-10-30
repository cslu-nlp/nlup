from distutils.core import setup


setup(name="nlup",
      version="0.1",
      description="Core libraries for natural language processing, with perceptron classifiers",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      url="http://github.com/cslu-nlp/nlup/",
      install_requires=["jsonpickle >= 0.8.0"],
      packages=["nlup"])
