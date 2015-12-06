# PyCEbox
Python Individual Conditional Expectation Plot Toolbox

## Development

For easy development and prototyping using IPython notebooks, there is a Vagrant envirnoment included.  To run an IPython notebook with access to your development version of `pycebox`, follow these steps.

1. Make sure [VirtualBox](https://www.virtualbox.org/wiki/Downloads) is installed (or change the Vagrant provider in [`Vagrantfile`](./Vagrantfile)).
2. Provision the virtual machine with `vagrant up`.  This will take some time, as it install `pycebox`'s dependencies.
3. Access the virtual machine with `vagrant ssh`.
4. Run an IPython notebook server with `PYTHONPATH=${PYTHONPATH}:/vagrant ipython notebook --no-browser --ip='*'`.

You should now be able to access the notebook server at `localhost:8000/tree`.
