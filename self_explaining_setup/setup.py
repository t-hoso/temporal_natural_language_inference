import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='Self_Explaining_Structures_Improve_NLP_Models',
    version='0.0.1',
    description='package for Self_Explaining_Structures_Improve_NLP_Models',
    packages=find_packages(exclude=('check')),
    install_requires=["numpy", "torch", "transformers"],
)