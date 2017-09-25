#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'PyYAML==3.12',
    'future==0.16.0',
    'scipy>=0.17.0',
    'numpy>=1.13.0',
    'matplotlib>=1.3.1',
    'mock==2.0.0',
]

setup_requirements = [
    # TODO(ljvmiranda921): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'PyYAML==3.12',
    'future==0.16.0',
    'scipy>=0.17.0',
    'numpy>=1.13.0',
    'matplotlib>=1.3.1',
    'mock==2.0.0',
]

setup(
    name='pyswarms',
    version='0.1.7',
    description="A Python-based Particle Swarm Optimization (PSO) library.",
    long_description=readme + '\n\n' + history,
    author="Lester James V. Miranda",
    author_email='ljvmiranda@gmail.com',
    url='https://github.com/ljvmiranda921/pyswarms',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='pyswarms',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
