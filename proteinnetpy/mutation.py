"""
Module containing functions for mutating ProteinNetRecords and feeding that data into further computations (e.g. Tensorflow).
These functions are fairly specific so may often be better used as inspiration to build users own solutions.
"""
import random
import logging

import numpy as np

from .data import LabeledFunction
from .maths import softmin_sample, softmax_sample

__all__ = ["ProteinNetMutator", "sequence_mutator", "per_position_mutator",
           "sample_deleterious", "sample_neutral"]

class ProteinNetMutator(LabeledFunction):
    """
    Map function generating mutated records.


    Apply a mutator function to a ProteinNet record and return the mutated sequence. This is a LabeledFunction that can be used to generate a TensorFlow Dataset. This setup is fairly specific to your downstream model design, so it will often be more useful to use it as a base to create an alternate implementation.

    Returns are in the form:
        ([wt_seq], mut_seq, [phi, psi, chi1]), label, [weights]

    Attributes
    ----------
    wildtype      : bool
        Outputs wildtype as well as mutant sequence.
    phi           : bool
        Outputs Phi backbone angles.
    psi           : bool
        Outputs Psi backbone angles.
    chi           : bool
        Outputs rotamer angles.
    mutator       : function
        Mutator function taking a ProteinNetRecord and returning the sampled variants and their deleteriousness. The return format depends on `per_position`. If per_position=False must return a tuple with the mutated sequence index array and whether it is deleterious (1/0). If per_position=True must return a tuple with mutant_seq, deleterious_inds, neutral_inds arrays.
    kwargs        : dict
        Keyword arguments passed to the mutator function.
    encoding      : dict
        Encoding mapping alphabetically encoded integer indeces to a new scheme.
    weights       : list
        List of float weights for WT, Deleterious and Neutral variants when mutating per position.
    func          : function
        Function applied when the class is called. This is a mutator applied to the whole sequence or per position derived from the initialisation parameters.
    output_shapes, output_types : tuple
        Tuple of output shapes and types (see data.LabeledFunction for details)
    """
    def __init__(self, mutator, per_position=False, include=('wt',),
                 weights=(0, 1, 1), encoding=None, **kwargs):
        """
        Initialise the mutator.

        Parameters
        ----------
        mutator       : function
            Mutator function taking a ProteinNetRecord and returning the sampled variants and their deleteriousness. The return format depends on `per_position`. If per_position=False must return a tuple with the mutated sequence index array and whether it is deleterious (1/0). If per_position=True must return a tuple with mutant_seq, deleterious_inds, neutral_inds arrays.
        per_position  : bool
            Return deleteriousness for each position rather than the entire sequence.
        include:      : sequence_like including some of {"wt", "phi", "psi", "chi1"}
            Variables to return for computation, alongside mutant sequence.
        weights       : list
            weightings for [wt, deleterious, neutral] variants when processing per residue variants
        encoding      : dict
            Optional dictionary mapping alphabetically encoded AA integer indeces to a new scheme (e.g. that used in UniRep).
        **kwargs      : dict
            arguments passed on to mutator
        """
        self.wildtype = 'wt' in include
        self.phi = 'phi' in include
        self.psi = 'psi' in include
        self.chi = 'chi1' in include

        self.mutator = mutator
        self.kwargs = kwargs

        self.encoding = encoding
        self.weights = weights

        if per_position:
            self.func = self._per_position_func
        else:
            self.func = self._whole_seq_func

        # Calculate output shape/type (Mutant sequence is always output)
        self.output_shapes = [[None]]
        self.output_types = ['int32']

        if self.wildtype:
            self.output_shapes.insert(0, [None])
            self.output_types.insert(0, 'int32')

        if self.phi:
            self.output_shapes.append([None])
            self.output_types.append('float32')

        if self.psi:
            self.output_shapes.append([None])
            self.output_types.append('float32')

        if self.chi:
            self.output_shapes.append([None])
            self.output_types.append('float32')

        if per_position:
            self.output_shapes = (tuple(self.output_shapes), [None, 1], [None])
            self.output_types = (tuple(self.output_types), 'int32', 'int32')
        else:
            self.output_shapes = (tuple(self.output_shapes), [])
            self.output_types = (tuple(self.output_types), 'int32')

    def _per_position_func(self, record):
        """
        Function producing per position mutants and labels from a ProteinNetRecord

        Parameters
        ----------
        record : ProteinNetRecord
            Record to mutate.
        """
        mut_seq, deleterious, neutral = self.mutator(record, **self.kwargs)

        mut_weights = np.full(deleterious.shape, self.weights[0])
        mut_weights[deleterious] = self.weights[1]
        mut_weights[neutral] = self.weights[2]

        if self.encoding is not None:
            mut_seq = np.array([self.encoding[x] for x in mut_seq])

        if self.wildtype:
            if self.encoding is not None:
                wt_seq = np.array([self.encoding[x] for x in record.primary_ind])
            else:
                wt_seq = record.primary_ind

            output = [wt_seq + 1, mut_seq + 1] # +1 to leave 0 free for padding
        else:
            output = [mut_seq + 1]

        if self.phi:
            output.append(record.rama[1])

        if self.psi:
            output.append(record.rama[2])

        if self.chi:
            output.append(record.chi)

        return tuple(output), deleterious.reshape((len(deleterious), 1)), mut_weights

    def _whole_seq_func(self, record):
        """
        Function applying mutator to a record and producing mutant data and a single label for the sequence from a ProteinNetRecord.

        Parameters
        ----------
        record : ProteinNetRecord
            Record to mutate.
        """
        mut_seq, label = self.mutator(record, **self.kwargs)

        if self.encoding is not None:
            mut_seq = np.array([self.encoding[x] for x in mut_seq])

        if self.wildtype:
            if self.encoding is not None:
                wt_seq = np.array([self.encoding[x] for x in record.primary_ind])
            else:
                wt_seq = record.primary_ind

            output = [wt_seq + 1, mut_seq + 1] # +1 to leave 0 free for padding
        else:
            output = [mut_seq]

        if self.phi:
            output.append(record.rama[1])

        if self.psi:
            output.append(record.rama[2])

        if self.chi:
            output.append(record.chi)

        return tuple(output), label

def sequence_mutator(record, p_deleterious=0.5, max_mutations=3,
                     max_deleterious=0.01, min_neutral=0.1):
    """
    Generate mutated sequences from a ProteinNetRecord with a few deleterious or neutral variants.

    Generate mutated sequences from a ProteinNetRecord with a few deleterious and/or neutral variants. First randomly choose to generate a deleterious or neutral sequence then sample some of the corresponding variant types based on the records MSA frequencies.

    Parameters
    ----------
    record          : ProteinNetRecord
        Record to mutate.
    p_deleterious   : float
        Probability of returning a deleterious set of variants.
    max_mutations   : int
        Maximum number of mutations to make.
    max_deleterious : float
        Maximum MSA frequency for a variant to be considered deleterious.
    min_neutral     : float
        Minimum MSA frequency for a variant to be considered neutral.

    Returns
    -------
    tuple
        Tuple of the format (seq, deleterious). The first entry is the mutated amino acid sequence, encoded with integer indeces and the second is 1 if the sequence is deleterious and 0 if neutral.
    """
    deleterious = int(random.random() < p_deleterious)
    seq = record.primary_ind.copy()

    num_muts = random.randint(1, max_mutations)

    if deleterious:
        positions, subs = sample_deleterious(num=num_muts, wt_seq=seq,
                                             pssm=record.evolutionary,
                                             max_freq=max_deleterious)
    else:
        positions, subs = sample_neutral(num=num_muts, wt_seq=seq,
                                         pssm=record.evolutionary,
                                         min_freq=min_neutral)

    seq[positions] = subs

    return seq, deleterious

def per_position_mutator(record, max_deleterious=2, max_neutral=4,
                         max_deleterious_freq=0.01, min_neutral_freq=0.1):
    """
    Generate mutated sequences from ProteinNetRecords with labels identifying deleterious and neutral mutations.

    Generate mutated sequences from ProteinNetRecords with labels identifying where deleterious and neutral mutations have been made.
    Will always generate at least one variant.

    Parameters
    ----------
    record               : ProteinNetRecord
        Record to mutate.
    max_deleterious      : int
        Maximum number of deleterious variants to make.
    max_neutral          : int
        Maximum number of neutral variants to make.
    max_deleterious_freq : float
        Maximum MSA frequency for a variant to be considered deleterious.
    min_neutral_freq     : float
        Minimum MSA frequency for a variant to be considered neutral.

    Returns
    -------
    tuple
        Tuple of the format seq, deleterious, neutral. The first entry is the mutated sequence, the second a list of positions with deleterious variants and the third a list of positions with neutral variants.
    """
    seq = record.primary_ind.copy()

    # Masks identifying where deletious and neutral mutations are made
    deleterious = np.zeros(seq.shape, dtype=int)
    neutral = np.zeros(seq.shape, dtype=int)

    # Sample number of substitutions [deleterious, neutral]
    num_deleterious = random.randint(0, max_deleterious)
    num_neutral = random.randint(0 if num_deleterious else 1, max_neutral)

    pos = None # Need a mask for sample_neutral if not making any del subs
    del_subs = 0 # Number of subs actually made
    neut_subs = 0
    if num_deleterious:
        try:
            pos, subs = sample_deleterious(num=num_deleterious,
                                           wt_seq=record.primary_ind,
                                           pssm=record.evolutionary,
                                           max_freq=max_deleterious_freq)

            seq[pos] = subs
            deleterious[pos] = 1
            del_subs = len(pos)

        except ValueError:
            pass

    if num_neutral:
        try:
            pos, subs = sample_neutral(num=num_neutral,
                                       wt_seq=record.primary_ind,
                                       pssm=record.evolutionary,
                                       min_freq=min_neutral_freq,
                                       mask=pos)

            seq[pos] = subs
            neutral[pos] = 1
            neut_subs = len(subs)

        except ValueError:
            pass

    if not del_subs and not neut_subs:
        raise ValueError('No valid mutations could be made')

    return seq, deleterious, neutral

def sample_deleterious(num, pssm, wt_seq, max_freq=0.025, mask=None):
    """
    Sample deleterious mutations from a MSA frequency matrix.

    Randomly choose a selection of deleterious variants from a MSA frequency matrix.

    Parameters
    ----------
    num      : int
        Number of mutations to make.
    pssm     : float ndarray (20, N)
        MSA frequency matrix to determine neutral and deleterious variants.
    wt_seq   : int ndarray (N,)
        WT sequence of the protein (as int indeces corresponding to the MSA matrix rows).
    max_freq : float
        Maximum frequency considered deleterious.
    mask     : int array_like
        Array of positions not to mutate.

    Returns
    -------
    tuple
        Numpy array of position indeces chosen and an array of the alternate amino acid in each position (as MSA row indeces).
    """
    if num == 0:
        raise ValueError('num must be > 0')

    # Select valid positions
    pssm[wt_seq, np.arange(pssm.shape[1])] = 0
    positions = np.nonzero(np.sum(pssm < max_freq, 0))[0]
    if not mask is None:
        positions = positions[~np.isin(positions, mask)]

    if positions.size == 0:
        raise ValueError('No valid neutral substitutions')

    if num > positions.size:
        logging.warning(('Too few positions to sample %s deleterious mutations, '
                         'sampling as many as possible (%s)'), num, positions.size)
        num = positions.size

    positions = np.random.choice(positions, replace=False, size=num)

    # Sample and make substitutions
    pssm = pssm[:, positions]
    substitutions = np.apply_along_axis(lambda x: softmin_sample(x, max_value=max_freq), 0, pssm)
    return positions, substitutions

def sample_neutral(num, pssm, wt_seq, min_freq=0.025, mask=None):
    """
    Sample deleterious mutations froma pssm

    Parameters
    ----------
    num      : int
        Number of mutations to make.
    pssm     : float ndarray (20, N)
        MSA frequency matrix to determine neutral and deleterious variants.
    wt_seq   : int ndarray (N,)
        WT sequence of the protein (as int indeces corresponding to the MSA matrix rows).
    min_freq : float
        Minimum frequency considered neutral.
    mask     : int array_like
        Array of positions not to mutate.

    Returns
    -------
    tuple
        Numpy array of position indeces chosen and an array of the alternate amino acid in each position (as MSA row indeces).
    """
    if num == 0:
        return None, None

    # Select valid positions
    pssm[wt_seq, np.arange(pssm.shape[1])] = 0
    positions = np.nonzero(np.sum(pssm > min_freq, 0))[0]
    if not mask is None:
        positions = positions[~np.isin(positions, mask)]

    if positions.size == 0:
        raise ValueError('No valid neutral substitutions')

    if num > positions.size:
        logging.warning(('Too few positions to sample %s deleterious mutations, '
                         'sampling as many as possible (%s)'), num, positions.size)
        num = positions.size

    positions = np.random.choice(positions, replace=False, size=num)

    # Sample and make substitutions
    pssm = pssm[:, positions]
    substitutions = np.apply_along_axis(lambda x: softmax_sample(x, min_value=min_freq), 0, pssm)
    return positions, substitutions
