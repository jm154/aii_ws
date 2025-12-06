from setuptools import find_packages
from setuptools import setup

setup(
    name='wall_follow',
    version='0.0.0',
    packages=find_packages(
        include=('wall_follow', 'wall_follow.*')),
)
