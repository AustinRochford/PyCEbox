FROM jupyter/scipy-notebook

MAINTAINER Austin Rochford <austin.rochford@gmail.com>

USER $NB_USER 

RUN pip3 install pytest hypothesis hypothesis-numpy

# to fix a bug in QT support; hopefully this is not necessary long-term
RUN conda install --quiet --yes icu=56.1

# Import matplotlib the first time to build the font cache.
RUN python -c "import matplotlib.pyplot"

ENV PYTHONPATH $PYTHONPATH:"$HOME"/pycebox
