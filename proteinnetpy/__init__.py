"""
A python library for working with ProteinNet data (see https://github.com/aqlaboratory/proteinnet)
"""

import logging

from proteinnetpy import parser
from proteinnetpy import record
from proteinnetpy import data
from proteinnetpy import maths
from proteinnetpy import mutation

__all__ = [i for j in (parser.__all__, record.__all__,
                       data.__all__, maths.__all__,
                       mutation.__all__) for i in j]

try:
    from proteinnetpy import tfdataset
    __all__.append(tfdataset.__all__)
except ModuleNotFoundError:
    logging.warning(("Tensorflow >=2.0 not available, install to access ProteinNet "
                     "tensorflow Datasets"))

