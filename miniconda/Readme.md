# Tutorial on how to install miniconda



#### Miniconda is a light weight python package manager for easy management of modules between projects

To install Miniconda, go to https://docs.conda.io/en/latest/miniconda.html and download the installer corresponding to your platform. 

Note that you should only download either the Linux or the OSX version and choose python3.7

After the download has completed, run the installer by typing

`sh Miniconda3-latest-MacOSX-x86_64.sh` (for Mac), or

`sh Miniconda3-latest-Linux-x86_64.sh` (for Linux).

Then follow the instructions on screen to finish the installation.



#### Introduction to using Miniconda

To create a new environment, type

`conda create --name ENV_NAME python=3.6`

To activate the environment, type

`source activate ENV_NAME`

While inside an environment, you can type `which python` to check that the python binaries have changed to the miniconda one. `which pip` tells you that pip is now using the miniconda version. You can still install packages using pip. 

To deactivate the environment, type

`source deactivate`



#### TA's environment

For each HW, we will release a Miniconda environment for you to test your code.

If it works in the environment, it should work in our machines.

To install the environment, type

`conda create --name <env> --file TA_ENV_FILE`

