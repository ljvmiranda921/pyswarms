#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

# Import modules
from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    dev_requirements = f.read().splitlines()


setup(
    name="pyswarms",
    version="0.4.0",
    description="A Python-based Particle Swarm Optimization (PSO) library.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Lester James V. Miranda",
    author_email="ljvmiranda@gmail.com",
    url="https://github.com/ljvmiranda921/pyswarms",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=requirements,
    tests_require=dev_requirements,
    extras_require={"test": dev_requirements},
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
)
