#!/bin/bash

export PATH=$PATH:/usr/local/bin

sudo yum update -y

# 2. Install Python 3 and pip
#sudo yum install -y python3
#
## 3. Ensure pip is installed
#python3 -m ensurepip --upgrade
#
## 4. (Optional) Create a symlink for `pip`
#sudo ln -sf /usr/bin/pip3 /usr/bin/pip

#sudo yum install -y git htop unzip bzip2 zip tar rsync emacs-nox xsltproc java-11-openjdk-devel cmake gcc gcc-c++ lapack-devel lz4-devel

if grep isMaster /mnt/var/lib/info/instance.json | grep true; then
	# Master node: Install all
	WHEELS="pyserial
	oauth
	argparse
	parsimonious
	wheel
	pandas
	utils
	ipywidgets
	numpy
	scipy
	bokeh
	requests
	boto3
	selenium
	pillow
	jupyterlab"
else
	# Worker node: Install all but jupyter lab
	WHEELS="pyserial
	oauth
	argparse
	parsimonious
	wheel
	pandas
	utils
	ipywidgets
	numpy
	scipy
	bokeh
	requests
	boto3
	selenium
	pillow"
fi

for WHEEL_NAME in $WHEELS
do
	pip install $WHEEL_NAME
done
