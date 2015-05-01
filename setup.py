from setuptools import setup


setup(name="nlup",
      include_package_data=True,
      install_requires=['jsonpickle==0.9.0'],
      version="0.5",
      description="Core libraries for natural language processing",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      # url="http://github.com/cslu-nlp/nlup/",
      install_requires=["jsonpickle >= 0.9.0"],
      packages=["nlup"],
      )
