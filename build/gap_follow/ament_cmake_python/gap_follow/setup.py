from setuptools import find_packages
from setuptools import setup

setup(
    name='gap_follow',
    version='0.0.0',
    packages=find_packages(
        include=('gap_follow', 'gap_follow.*')),
)
