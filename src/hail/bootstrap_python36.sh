#!/bin/bash
set -e

# This file is available in ultima bucket: s3://ultimagen-gil-hornung/hail_on_emr/bootstrap_python3.sh

export PATH=$PATH:/usr/local/bin

cd $HOME
mkdir -p $HOME/.ssh/id_rsa
#sudo yum install python36 python36-devel python36-setuptools -y
#sudo easy_install pip
#sudo python3 -m pip install --upgrade pip

sudo yum update -y

## 2. Install Python 3 and pip
#sudo yum install -y python3
#
## 3. Ensure pip is installed
#python3 -m ensurepip --upgrade
#
## 4. (Optional) Create a symlink for `pip`
#sudo ln -sf /usr/bin/pip3 /usr/bin/pip

echo "bootstrap_python3 start"

if grep isMaster /mnt/var/lib/info/instance.json | grep true; then
    sudo yum install g++ cmake git -y
#    sudo yum install gcc72-c++ -y # Fixes issue with c++14 incompatibility in Amazon Linux
#    sudo yum install lz4 lz4-devel -y # Fixes issue of missing lz4
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
	python-magic
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
	python-magic"
fi

pip install "prompt-toolkit<3.0.39,>=3.0.24"
pip install "python-dateutil<=2.9.0"

echo "bootstrap_python3 start install"

for WHEEL_NAME in $WHEELS
do
	pip install $WHEEL_NAME
done

echo "bootstrap_python3 done"
# Ref: https://www.codeammo.com/article/install-phantomjs-on-amazon-linux/
# Install phantomjs-2.1.1 for bokeh.export_png
# yum install fontconfig freetype freetype-devel fontconfig-devel libstdc++
# wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
# sudo mkdir -p /opt/phantomjs
# bzip2 -d phantomjs-2.1.1-linux-x86_64.tar.bz2
# sudo tar -xvf phantomjs-2.1.1-linux-x86_64.tar \
#     --directory /opt/phantomjs/ --strip-components 1
# sudo ln -s /opt/phantomjs/bin/phantomjs /usr/bin/phantomjs

sudo yum update -y # It has to be at the end so it does not interfere with other yum installations