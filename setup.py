#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

requirements = [
    "PyYAML==3.13",
    "future==0.16.0",
    "scipy>=0.17.0",
    "numpy>=1.13.0",
    "matplotlib>=1.3.1",
    "mock==2.0.0",
    "pytest==3.6.4",
    "attrs==18.1.0",
    "pre-commit",
]

setup_requirements = [
    # TODO(ljvmiranda921): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    "PyYAML==3.13",
    "future==0.16.0",
    "scipy>=0.17.0",
    "numpy>=1.13.0",
    "matplotlib>=1.3.1",
    "mock==2.0.0",
    "pytest==3.6.4",
    "attrs==18.1.0",
    "pre-commit",
]

setup(
    name="pyswarms",
    version="0.3.1",
    description="A Python-based Particle Swarm Optimization (PSO) library.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Lester James V. Miranda",
    author_email="ljvmiranda@gmail.com",
    url="https://github.com/ljvmiranda921/pyswarms",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords="pyswarms",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
