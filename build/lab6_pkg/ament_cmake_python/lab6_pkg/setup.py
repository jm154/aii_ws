from setuptools import find_packages
from setuptools import setup

setup(
    name='lab6_pkg',
    version='0.0.0',
    packages=find_packages(
        include=('lab6_pkg', 'lab6_pkg.*')),
)
