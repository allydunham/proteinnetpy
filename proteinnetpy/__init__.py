"""
A python library for working with ProteinNet data (see https://github.com/aqlaboratory/proteinnet)
"""
__version__ = "0.1.0"

import logging

from proteinnetpy import data
from proteinnetpy import maps
from proteinnetpy import maths

try:
    from proteinnetpy import datasets
except ModuleNotFoundError:
    logging.warning(("Tensorflow >=2.0 not available, install to access ProteinNet "
                     "tensorflow Datasets"))
