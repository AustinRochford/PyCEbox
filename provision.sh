#!/bin/bash

#add-apt-repository ppa:marutter/rrutter
apt-get update

# Install IPython
apt-get install -y python-pip python2.7-dev libzmq-dev
pip install "ipython[notebook]==3.2.1"

# Install numpy and scipy
apt-get install -y gfortran libblas-dev liblapack-dev
pip install numpy scipy

# Install matplotlib, Seaborn
apt-get install -y libpng12-dev libfreetype6-dev
ln -s /usr/include/freetype2/ft2build.h /usr/include/
pip install matplotlib seaborn

# Install pandas and scikit-learn
pip install pandas
pip install scikit-learn
