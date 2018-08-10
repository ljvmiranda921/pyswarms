#!/usr/bin/env bash

export DEBIAN_FRONTEND=noninteractive

# Update sources
sudo apt-get update -y
 
# Git
sudo apt-get install git

# Python
sudo apt-get install  -y python-pip python-dev build-essential 

# PyYaml
sudo apt-get update
sudo apt-get install python-yaml

# Future

sudo apt-get update
pip install future

# Scipy & Numpy
sudo apt-get update
sudo apt-get install python-numpy python-scipy

# matplotlib
sudo apt-get update
sudo apt-get install python-matplotlib

# Mock
sudo apt-get update
pip install mock

# pytest
sudo apt-get update
pip install -U pytest

# attrs
sudo apt-get update
pip install attrs

# AWS Cli
sudo apt-get update
pip install awscli

# Vim
sudo apt-get install vim -y