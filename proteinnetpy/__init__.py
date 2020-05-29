"""
A python library for working with ProteinNet data (see https://github.com/aqlaboratory/proteinnet)
"""
__version__ = "0.2.0"

import logging

from proteinnetpy import parser
from proteinnetpy import record
from proteinnetpy import data
from proteinnetpy import maths
from proteinnetpy import mutation

try:
    from proteinnetpy import tfdataset
except ModuleNotFoundError:
    logging.warning(("Tensorflow >=2.0 not available, install to access ProteinNet "
                     "tensorflow Datasets"))
