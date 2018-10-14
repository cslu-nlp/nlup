from setuptools import setup

description = "Core libraries for natural language processing",

# To keep the docs pretty, use pandoc to "compile" the rst version of the README file
# before uploading to cheeseshop:
#
#     pandoc README.md -o README.rst
try:
  long_description = open('README.rst', 'r').read()
except:  # (IOError, ImportError, OSError, RuntimeError):
  long_description = description
  print('WARNING: Unable to find or read README.rst.')

setup(name="nlup",
      include_package_data=True,
      version="0.7",
      description=description,
      author="Kyle Gorman",
      author_email="kylebgorman@gmail.com",
      url="http://github.com/cslu-nlp/nlup/",
      install_requires=["jsonpickle >= 0.9.0", "nltk >= 3.1"],
      packages=["nlup"],
      keywords=["nlp", "natural language processing", "text", "text processing", "ai", "artificial intelligence", "neural net", "perceptron", "data", "science", "statistics", "data science", "math", "machine learning", "computer science", "information theory"],
      classifiers=[
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.5",
          "Development Status :: 5 - Production/Stable",
          "Environment :: Other Environment",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Text Processing :: Linguistic",
          "Topic :: Text Processing :: Filters",
          "Topic :: Text Processing :: General",
          "Topic :: Text Processing :: Indexing",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          ],
      )
