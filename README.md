# Dataset maker for GOTM
Make a dataset of GOTM simulations with idealised surface forcing for ML training and testing.

# Installing GOTM 

Step 1: Install GOTM (conda/mamba env to build)

a. Install miniconda: https://docs.rdhpcs.noaa.gov/software/python/miniforge.html

b. Make directory for GOTM:  
mkdir GOTM  
cd GOTM  
git clone https://github.com/gotm-model/code gotm-code  
(optional)  
cd ..  
git clone https://github.com/gotm-model/cases gotm-cases  

c. Make conda environment for GOTM:  
conda create -n gotm-env -c conda-forge fortran-compiler netcdf-fortran cmake  
conda activate gotm-env  

d. Build GOTM:  
(In GOTM directory)  
mkdir gotm-install  
mkdir build  
cd build  
cmake ../gotm-code -DGOTM_USE_FABM=off -DCMAKE_INSTALL_PREFIX=/absolute/path/to/GOTM/gotm-install  
make -j$(nproc)  
make install  

e. Activate GOTM:  
export PATH="/absolute/path/to/GOTM/gotm-install/bin:$PATH"  

f. Verify installation (and optionally, run a test case):  
gotm --version  
cd ../gotm-cases/ows_papa/  
gotm  

# Using the maker

In run.py, define _M_ temperature gradients, _N_ wind stresses, _J_ latitudes and _K_ heat fluxes to include as forcing for the cases. There will be _MNJK_ cases in total. The training set name will be the name of the directory where all output is saved. Case names run from _case\_1_ to _case\_MNJK_ and they are strings in the dictionary of cases, which is kind of silly and makes things unnecessarily hard. (Sorry about that. Early bad choice.) All case specs are saved as a json in the root directory. In the output directory, each case has a separate folder, where the edited setup yaml file is saved, as well as netcdf files of output, restart, and the initial temperature profile.

The source_yaml directory contains GOTM setup file where the desired turbulence settings and other forcings can be defined. This file will be copied into each case folder and edited by the GOTM runner.

GOTM runner is available on a regular grid (pre-defined vertical and temporal grid spacing in the source yaml file) or on a _f_ and _u\_star_ grid (the amount of inertial oscillations and depth layers are pre-defined and equal in each case, where total depth and total time elapsed are variable).

Feature retreiver takes the case dictionary and the dataset directory as inputs, as well as the grid settings, and can retreive a range of features from the output files. There are some messy functions here, so use caution when retreiving the features. Definitely could refactor this one...

Utility contains all helper functions in one big file as God intended. 

TRY OUT SMALL CASE SUBSET FIRST because stopping 1000+ cases is annoying. 

using_feature_retreiver.ipynb demonstrates how to extract data from the cases.

I am open to constructive criticism (◡‿◡✿) (◕ᴗ◕✿)
