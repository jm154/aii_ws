from setuptools import find_packages
from setuptools import setup

setup(
    name='pure_pursuit',
    version='0.0.0',
    packages=find_packages(
        include=('pure_pursuit', 'pure_pursuit.*')),
)
