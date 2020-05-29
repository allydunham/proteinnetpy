"""
Classes to map function over ProteinNet datasets
"""
from abc import ABC

import numpy as np
from .data import record_parser, PROTEINNET_FIELDS

# Factory pattern class for creating maps
class ProteinNetMap:
    """
    Iterate over ProteinNetRecords, applying a function to each record, with the ability
    to repeatedly loop over the data when using a stochastic function to get different
    results on each pass.

    path: Path to file to read from
    data: Iterable of ProteinNetRecords
    func: Function to apply to each record
    filter_func: Function returning truthy values indicating whether to keep each record
    preload: Bool indicating whether file data should be read into memory
    static: Result of map is static, so precalculate and cache all results
    """
    @staticmethod
    def from_file(path, func=None, filter_func=None, preload=True,
                  filter_errors=True, static=False, **kwargs):
        """
        Create a ProteinNetMap from a file
        """
        if static:
            return ProteinNetStaticMap(record_parser(path, **kwargs), func,
                                       filter_func, filter_errors)
        if preload:
            data = list(record_parser(path, **kwargs))
            return ProteinNetMap.from_data(data, func, filter_func, filter_errors)

        return ProteinNetStreamMap(path, func, filter_func, filter_errors, **kwargs)

    @staticmethod
    def from_data(data, func=None, filter_func=None, filter_errors=True, static=False):
        """
        Create a ProteinNetMap from in memory data
        """
        if static:
            return ProteinNetStaticMap(data, func, filter_func, filter_errors)
        else:
            return ProteinNetDataMap(data, func, filter_func, filter_errors)

class ProteinNetStaticMap(ProteinNetMap):
    """
    ProteinNetMap opperating on an iterable of data, precalculating the result of func
    on each record, as it is assumed to be constant.

    This is a convenience class allowing downstream code expecting a ProteinNetMap
    to function while avoiding recalculating constant values when compute time is
    more valuable than memory.
    """
    def __init__(self, data, func=None, filter_func=None, filter_errors=True):
        if func is None:
            func = lambda x: x

        self.func = func
        self.filter_func = filter_func
        self.filter_errors = filter_errors
        self.data = []

        for rec in data:
            if self.filter_func is not None and not self.filter_func(rec):
                continue

            try:
                self.data.append(func(rec))
            except ValueError as err:
                if self.filter_errors:
                    logging.warning('Skipping error: %s', err)
                    continue
                raise err

    def __len__(self):
        return len(self.data)

    def generate(self):
        """
        Yield results of func applied to each record in data
        """
        for x in self.data:
            yield x

class ProteinNetDataMap(ProteinNetMap):
    """
    ProteinNetMap opperating on in memory data
    """
    def __init__(self, data, func=None, filter_func=None, filter_errors=True):
        if func is None:
            func = lambda x: x

        self.data = data
        self.func = func
        self.filter_errors = filter_errors

        if filter_func is not None:
            self.data = [x for x in self.data if filter_func(x)]

    def __len__(self):
        return len(self.data)

    def generate(self):
        """
        Yield results of func applied to each record in data
        """
        for rec in self.data:
            try:
                yield self.func(rec)
            except ValueError as err:
                if self.filter_errors:
                    logging.warning('Skipping error: %s', err)
                    continue
                raise err

class ProteinNetStreamMap(ProteinNetMap):
    """
    ProteinNetMap opperating on data streamed from files
    """
    def __init__(self, path, func=None, filter_func=None, filter_errors=True,
                 max_len=float('inf'), excluded_fields=None, normalise_angles=False,
                 profiles=None):
        if func is None:
            func = lambda x: x

        if filter_func is None:
            filter_func = lambda x: True

        self.path = path
        self.func = func
        self.filter_func = filter_func
        self.max_len = max_len
        self.excluded_fields = excluded_fields
        self.normalise_angles = normalise_angles
        self.filter_errors = filter_errors
        self.profiles = profiles

    def __len__(self):
        return sum(self.filter_func(r) for r in record_parser(self.path, max_len=self.max_len,
                                                              excluded_fields=PROTEINNET_FIELDS,
                                                              profiles=self.profiles))

    def generate(self):
        """
        Yield modified records from the file, passed through func
        """
        for rec in record_parser(path=self.path, max_len=self.max_len,
                                 excluded_fields=self.excluded_fields,
                                 normalise_angles=self.normalise_angles,
                                 profiles=self.profiles):
            if self.filter_func(rec):
                try:
                    yield self.func(rec)
                except ValueError as err:
                    if self.filter_errors:
                        logging.warning('Skipping error: %s', err)
                        continue
                    raise err

class ProteinNetMapFunction:
    """
    Labeled functions to map across ProteinNet data, including data on
    the function and it's output. In particular they store the output
    data type and shape for inputting into tf.Dataset objects
    """
    def __init__(self, func, output_types, output_shapes):
        self.func = func
        self.output_types = output_types
        self.output_shapes = output_shapes

    def __call__(self, record):
        return self.func(record)

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
