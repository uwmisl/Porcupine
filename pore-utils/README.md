# pore_utils

Utilites for manipulating raw nanopore current.

This is my library of ad hoc python modules which can be imported for use in other projects.

Within the context of Porcupine, I've only included the code for `smith_waterman`, so note that `pore_utils` is a slight misnomer in the absence of these other nanopore utilities that are irrelevant for Porcupine (e.g. accessors for bulk nanopore files).
    
## How to install
Pip install this package. I prefer development mode if changes will be made. This means even if you pull the repo or make changes, imported modules will use the code in this directory instead of the typical install directory.

    cd pore-utils
    pip install -e .

## How to import

    from pore_utils import smith_waterman
    
Then all the modules can be used like:

    raw_signal_utils.get_raw(f5)
