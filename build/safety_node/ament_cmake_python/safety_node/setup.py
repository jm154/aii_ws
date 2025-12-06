from setuptools import find_packages
from setuptools import setup

setup(
    name='safety_node',
    version='0.0.0',
    packages=find_packages(
        include=('safety_node', 'safety_node.*')),
)
