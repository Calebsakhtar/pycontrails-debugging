Set up a new conda environment to recreate the results:
conda create --name pycontrails-debug python=3.10
conda activate pycontrails-debug
python -m pip install "pycontrails[complete]"
python -m pip uninstall pycontrails
python -m pip install git+https://github.com/Calebsakhtar/pycontrails.git@ca525/latest
python -m pip install xarray==2023.10.0
conda install -c conda-forge eccodes

Note that installing "pycontrails[complete]" installs all of the dependencies. Pycontrails can then be removed,
and the debug version can be installed.

This debug version has disabled the following:
- Disabled logitude/latitude advection
- Enforcing of normal shear = -2e-3
- Enforcing of total shear = 2e-3
- Makes max lifespan 24 hrs
- Makes integration timestep 5 mins
- Disable wind shear enhancement
All the differences between the debug version and the main branch are available here: https://github.com/contrailcirrus/pycontrails/compare/main...Calebsakhtar:pycontrails:ca525/latest

The script "reproduce-bug.py" evaluates CoCiP over "pycontrails-flight.csv" at all custom met inputs available in "inputs/met".
Results are then saved to an outputs folder which is automatically created.
