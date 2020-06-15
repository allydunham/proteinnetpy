"""
Module containing methods and classes to work with ProteinNet Data
"""
import logging
import numpy as np
from .parser import record_parser

class ProteinNetDataset:
    """
    Container for ProteinNetRecords, supporting streaming from files or loading into memory

    path: Path to file to read from
    data: Iterable of ProteinNetRecords
    filter_function: filter function to apply to data, only Records where this returns True are kept
    preload: Bool indicating whether file data should be read into memory
    **kwargs: keyword arguments passed on to record_parser
    """
    def __init__(self, path=None, data=None, filter_func=None, preload=True, **kwargs):
        if (path is None and data is None) or (path is not None and data is not None):
            raise ValueError("One of 'path' or 'data' must be passed, not neither or both")

        self.path = path
        self.data = data
        self.filter_func = (lambda x: True) if filter_func is None else filter_func
        self.preload = preload if self.data is None else None # Preload is meaningless if using data
        self._parser = None
        self.parser_args = kwargs

        if self.preload and self.path is not None:
            self.data = [i for i in record_parser(self.path, **kwargs)]

        if self.data is not None:
            self.data = [i for i in self.data if self.filter_func(i)]

    def __len__(self):
        if self.data is None:
            return sum(1 for _ in self)
        return len(self.data)

    def __iter__(self):
        if self.data is None:
            self._parser = record_parser(self.path, **self.parser_args)
            return self
        return iter(self.data)

    def __next__(self):
        while True:
            record = next(self._parser)
            if self.filter_func(record):
                return record

    def __getitem__(self, index):
        if self.data is None:
            parser = record_parser(self.path, **self.parser_args)
            for _ in range(index + 1):
                record = next(parser)
            return record
        return self.data[index]

class LabeledFunction:
    """
    Functions labeled with output shape and data type for inputting into
    e.g. tf.Dataset objects that need to know types and shapes to initialise
    neural networks
    """
    def __init__(self, func, output_types, output_shapes):
        self.func = func
        self.output_types = output_types
        self.output_shapes = output_shapes

    def __call__(self, x)
        return self.func(x)

class ProteinNetMap:
    """
    Map a function over ProteinNetRecords, with the ability to repeatedly loop over the data when using a stochastic function to get different results on each pass. Setup to interface with
    neural network training loops expecting a generator.

    data:   A ProteinNetDataset
    func:   Function to apply to each record
    static: Result of map is constant, so precalculate and cache all results on initialisation
    """
    def __init__(self, data, func, filter_errors=True, static=False):
        self.data = data
        self.func = func
        self.filter_errors = filter_errors
        self._static = static

        if self._static:
            self.data = []
            for record in data:
                try:
                    self.data.append(func(record))
                except ValueError as err:
                    if self.filter_errors:
                        logging.warning('Skipping error: %s', err)
                        continue
                    raise err

    def __len__(self):
        if self._static:
            return len(self.data)
        return sum(1 for _ in self.generate())

    def __iter__(self):
        return self.generate()

    def generate(self):
        """
        Yield results of func applied to each record in data.
        Provided as an entrypoint for functions expecting a
        generator function to call to access data, for example
        tensorflow datasets.
        """
        if self._static:
            for x in self.data:
                yield x

        else:
            for record in self.data:
                try:
                    yield self.func(record)
                except ValueError as err:
                    if self.filter_errors:
                        logging.warning('Skipping error: %s', err)
                        continue
                    raise err

##########################################################
#################### Filter funcitons ####################
##########################################################

def combine_filters(*args):
    """
    Combine a series of filters into a single function
    """
    def func(rec):
        for filt in args:
            if not filt(rec):
                return False
        return True
    return func

def make_mask_filter(min_rama_prop=0, min_chi_prop=0, min_tertiary_prop=0):
    """
    Create basic filter functions based on record masks
    """
    if min_chi_prop == min_rama_prop == min_tertiary_prop == 0:
        raise ValueError('At least one mask proportion should be >0 to filter anything')

    def func(rec):
        rama = np.sum(rec.rama_mask)/len(rec.rama_mask) > min_rama_prop if min_rama_prop else True
        chi = np.sum(rec.chi_mask)/len(rec.chi_mask) > min_chi_prop if min_chi_prop else True
        tertiary = np.sum(rec.mask)/len(rec.mask) > min_tertiary_prop if min_tertiary_prop else True

        return rama and chi and tertiary

    return func

def profile_filter(rec):
    """
    Filter records without profiles
    """
    return rec.profiles is not None

def make_nan_filter(rama=True, chi=True, profiles=False):
    """
    Generate filter function checking the required fields for NaN
    """
    def func(record):
        if rama and np.isnan(np.min(record.rama)):
            return False
        if chi and np.isnan(np.min(record.chi)):
            return False
        if profiles and np.isnan(np.min(record.profiles)):
            return False
        return True
    return func
