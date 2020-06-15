"""
Module containing functions for mutating ProteinNetRecords and feeding that
data into further computations (e.g. Tensorflow)
"""
import random
import logging

import numpy as np

from .data import LabeledFunction
from .maths import softmin_sample, softmax_sample

class ProteinNetMutator(LabeledFunction):
    """
    Map function mutating records and outputting the mutant sequence
    in additions to other features. Produces the
    appropriate data for NN training with tensorflow.

    Returns are in the form:
        ([wt_seq], mut_seq, [phi, psi, chi1]), label, [weights]

    mutator: mutator function
    per_residue: whether mutants deleteriousness is tracked per sequence or per
                 mutant. Requires a compatible mutator returning a tuple of
                 mutant_seq, deleterious_inds, neutral_inds
    include: variables to return for computation, alongside mutant sequence.
             Options: (wt, phi, psi, chi1)
    weights: weightings for [wt, deleterious, neutral] variants when processing
             per residue variants
    encoding: Optional dictionary mapping alphabetically encoded AA indeces to a new
              scheme (e.g. that used in UniRep)
    **kwargs: arguments passed on to mutator
    """
    def __init__(self, mutator, per_position=False, include=('wt',),
                 weights=(0, 1, 1), encoding=None, **kwargs):
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
        Function producing per position labels
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
        Function producing a single label for the sequence
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
    Generate mutated sequences from ProteinNetRecords with a few deleterious or
    neutral variants, along with a label identifying them as deleterious (1) or neutral (0)
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

def per_residue_mutator(record, max_deleterious=2, max_neutral=4,
                        max_deleterious_freq=0.01, min_neutral_freq=0.1):
    """
    Geneate mutated sequences from ProteinNetRecords and return the variant sequence along
    with labels identifying deleterious and neutral positions. Will always generate at least
    one variant.
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
    Sample deleterious mutations froma pssm

    num: number of mutations to make
    pssm: PSSM in a np.array
    wt_seq: WT sequence of the protein
    max_freq: maximum frequency considered deleterious
    mask: positions to mask

    returns: np.array(positions), np.array(substitutions)
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

    num: number of mutations to make
    pssm: PSSM in a np.array
    wt_seq: WT sequence of the protein
    min_freq: minimum frequency considered neutral
    mask: positions to mask

    returns: np.array(positions), np.array(substitutions, as PSSM row indeces)
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
