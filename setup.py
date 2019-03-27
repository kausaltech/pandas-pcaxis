# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='pandas-pcaxis',
    packages=find_packages(),
    version='0.1',
    license='BSD',
    description='Pandas support for reading PC-Axis files',
    author='Juha Yrjölä',
    author_email='juha.yrjola@iki.fi',
    url='https://github.com/juyrjola/pandas-pcaxis',
    install_requires=[
        'pandas',
        'requests',
        'lark-parser',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
