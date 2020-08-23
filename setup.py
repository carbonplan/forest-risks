#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='forests',
    version=version,
    description=' ',
    author='carbonplan',
    author_email='tech@carbonplan.org',
    url='https://github.com/carbonplan/forests',
    packages=['forests'],
    install_requires=required,
    long_description='See ' + 'https://github.com/carbonplan/forests',
    license='MIT'
)
