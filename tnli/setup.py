import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='sentence_transe',
    version='0.0.1',
    description='package for Sentence TransE',
    author='Taishi Hosokawa',
    url='https://github.com/t-hoso/TransE.git',
    packages=find_packages(exclude=('tests', 'log', 'data')),
    install_requires=read_requirements(),
)