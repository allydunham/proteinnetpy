"""
Module containing methods and classes to work with ProteinNet Data
"""
import logging
import numpy as np
from .parser import record_parser

__all__ = ["ProteinNetDataset", "LabeledFunction", "ProteinNetMap", "combine_filters",
           "make_length_filter", "make_mask_filter", "make_id_filter", "profile_filter",
           "make_nan_filter"]

class ProteinNetDataset:
    """
    Iterable container for ProteinNet records.

    An iterable container for ProteinNet records, allowing looping over entries as record.ProteinNetRecord objects.
    It supports filtering, len() and indexing.
    Data is able to be loaded into memory or streamed during iteration to ballance speed and RAM usage.

    Attributes
    ----------
    path        : str
        Path to the ProteinNet file.
    data        : list or None
        List of record.ProteinNetRecord objects is preload=True else None and Records are loaded during iteration.
    filter_func :
        Truthy returning function that determines the records to keep in the dataset
    preload     : bool
        Data is loaded into memory rather than streamed on iteration.
    parser_args : dict
        Dictionary of keyword arguments to pass to the record parser.
    """
    def __init__(self, path=None, data=None, filter_func=None, preload=True, **kwargs):
        """
        Initialise the ProteinNetDataset.

        Parameters
        ----------
        path         : str
            Path to ProteinNet file to load
        data         : list
            List of ProteinNetRecords objects, allowing you to build a dataset from data that's already been loaded.
        filter_func  : Function
            Function returning a truthy value determining whether records are kept in the dataset. Returning true indicates a record should be kept.
        preload      : bool
            Load data into memory on initialisation, rather than streaming it on iteration.
        **kwargs     : Dict, optional
            Additional arguments to pass to parser.record_parser
        """
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
            record = None
            for _ in range(index + 1):
                record = next(parser)
            return record
        return self.data[index]

class LabeledFunction:
    """
    Function labeled with output shape and type.

    Functions labeled with output shape and data type for inputting into
    e.g. tf.Dataset objects that need to know types and shapes to initialise
    neural networks

    Attributes
    ----------
    func          : Function
        Function called.
    output_types  : tuple
        Potentially nested tuple of strings describing the data types output by the function. These should be in the form recognised by tf.as_dtype to use with TensorFlow datasets.
    output_shapes : tuple
        Potentially nested tuple of integer/None lists that describe the array/tensor shapes output by the function. These should be in the form recognised by tf.TensorShape to use with TensorFlow datasets.
    """
    def __init__(self, func, output_types, output_shapes):
        """
        Initialise the LabeledFunction

        Parameters
        ----------
        func          : Function
            Function called.
        output_types  : tuple
            Potentially nested tuple of strings describing the data types output by the function.
        output_shapes : tuple
            Potentially nested tuple of integer/None lists that describe the array/tensor shapes output by the function.
        """
        self.func = func
        self.output_types = output_types
        self.output_shapes = output_shapes

    def __call__(self, x):
        return self.func(x)

class ProteinNetMap:
    """
    Map a function over a ProteinNetDataset.

    Map a function over ProteinNetRecords, setup to interface with neural network training loops expecting a generator.
    It allows results to be stored to maximise speed on additional iterations or for the calculation to be repeated each time to minimise memory usage or generate novel results on each iteraton if the mapped function is stochastic.

    Attributes
    ----------
    data          : `ProteinNetDataset`
        `ProteinNetDataset` mapped over if `_static=False` or the calculated result if `_static=True`.
    func          : Function or `LabeledFunction`
        Function mapped over the records.
    filter_errors : bool
        Records raising an error are skipped with a warning, rather than stopping the map.
    _static       : bool
        The output is the same on each loop over the dataset
    """
    def __init__(self, data, func, filter_errors=True, static=False):
        """
        Initialise the ProteinNetMap.

        Parameters
        ----------
        data          : `ProteinNetDataset`
            `ProteinNetDataset` to map over.
        func          : Function or `LabeledFunction`
            Function to map over the records.
        filter_errors : bool
            Skip records that raise an error, rather than stopping calculation.
        static       : bool
            Calculate results once and store them for subsequent iterations rather than recalculating on each generation.
        """
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
        Apply `func` to each record in the dataset.

        Apply `func` to each record in the dataset, yielding the results as a generator.
        If `static=True` this simply maps over the precalculated results.
        This interfacet is provided as an entrypoint for functions expecting a generator function to call to access data, for example tensorflow datasets.

        Yields
        ------
        Variable
            Result of appling `self.func` to a `ProteinNetRecord`. See `x.func.output_types` and `x.func.output_shapes` to determine the types.
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

    Create a function the applies multiple filter functions to a record and returns `True` only if they all do.

    Parameters
    ----------
    *args : Functions
        Functions taking a `ProtienNetRecord` and returning True to keep it or False to filter it.

    Returns
    -------
    Function
        A function that applies all the functions in `*args` and returns the logical AND of their output.
    """
    def func(rec):
        for filt in args:
            if not filt(rec):
                return False
        return True
    return func

def make_length_filter(min_length=None, max_length=None):
    """
    Generate a filter function checking if a ProteinNetRecord is within length bounds.

    Generate a filter function checking if a ProteinNetRecord's sequence is within length bounds. Boundaries are open ended, meaning sequences at the boundaries are included.

    Parameters
    ----------
    min_length : int or None
        Minimum acceptable length. If None no lower bound is applied.
    max_length : int or None
        Maximum acceptable length. If None no upper bound is applied.

    Returns
    -------
    Function
        A function returning True if the min_length =< len(record) <= max_length.
    """
    min_length = 0 if min_length is None else min_length
    max_length = float('inf') if max_length is None else max_length
    def func(rec):
        length = len(rec)
        return length >= min_length and length <= max_length
    return func

def make_mask_filter(min_rama_prop=0, min_chi_prop=0, min_tertiary_prop=0):
    """
    Create a filter function requiring a minimum proportion of structural information to be present.

    Create a filter function requiring at least the given proportion of positions to have structural information for the given types.
    Ramachandran and Chi1 angles can be added to records using the `record.ProteinNetRecord.calculate_backbone_angles` method or the `add_angles_to_proteinnet` script.

    Parameters
    ----------
    min_rama_prop     : Float
        Minimum proportion of positions with backbone angle information present.
    min_chi_prop      : Float
        Minimum proportion of positions with Chi1 variable group angle information present.
    min_tertiary_prop : Float
        Minimum proportion of positions with tertiary structure coordinate information present.

    Returns
    -------
    Function
        A function returning True if all minimum proportions are met.
    """
    if min_chi_prop == min_rama_prop == min_tertiary_prop == 0:
        raise ValueError('At least one mask proportion should be >0 to filter anything')

    def func(rec):
        rama = np.sum(rec.rama_mask)/len(rec.rama_mask) > min_rama_prop if min_rama_prop else True
        chi = np.sum(rec.chi_mask)/len(rec.chi_mask) > min_chi_prop if min_chi_prop else True
        tertiary = np.sum(rec.mask)/len(rec.mask) > min_tertiary_prop if min_tertiary_prop else True

        return rama and chi and tertiary

    return func

def make_id_filter(pdb_ids, pdb_chains):
    """
    Generate a dataset filter only allowing specific PDB ID/Chains.

    Parameters
    ----------
    pdb_ids    : list
        List of PDB IDs to accept.
    pdb_chains : list
        List of PDB chains corresponding to the IDs in `pdb_ids`.

    Returns
    -------
    Function
        A function returning True for the given PDB ID/Chain and False otherwise.
    """
    ids = set([f"{pid.upper()}_{chn}" for pid, chn in zip(pdb_ids, pdb_chains)])
    def func(record):
        return f"{record.pdb_id}_{record.pdb_chain}" in ids
    return func

def profile_filter(rec):
    """
    Filter records without profiles.

    Filter records without profiles, which are additional numerical data associated with each position in the sequence.
    These are added by the user and can correspond to things like the output of language models like UniRep or AminoBert.

    Parameters
    ----------
    rec    : `record.Record`
        ProteinNetRecord to test.

    Returns
    -------
    bool
        True if profiles is not None, else False
    """
    return rec.profiles is not None

def make_nan_filter(rama=True, chi=True, profiles=False):
    """
    Generate filter function checking fields for NaN values.

    Generate filter function checking fields for NaN values, which can cause numerical issues for downstream analysis.
    Other Record data should not contain such values, but the function can be easily extended to check other features if necessary.

    Parameters
    ----------
    rama     : bool
        Check if any backbone angles are NaN.
    chi      : bool
        Check if any side chain Chi angles are NaN.
    profiles : bool
        Check if positional profiles contain any NaN values.

    Returns
    -------
    Function
        A function returning False if NaN values are present and True otherwise.
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
