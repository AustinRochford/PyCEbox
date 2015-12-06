#!/bin/bash

#add-apt-repository ppa:marutter/rrutter
apt-get update
apt-get install -y git 

# Install IPython
apt-get install -y python-pip python2.7-dev libzmq-dev
pip install "ipython[notebook]==3.2.1"

# Install matplotlib, Seaborn, and bokeh
apt-get install -y libpng12-dev libfreetype6-dev
ln -s /usr/include/freetype2/ft2build.h /usr/include/
pip install matplotlib

# Install pandas, patsy, statsmodels, scikit-learn and pymc3
pip install pandas
