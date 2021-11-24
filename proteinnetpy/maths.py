#!/usr/bin/env python3
"""
Module containing maths functionss for working with ProteinNet datasets
"""
import numpy as np
from Bio.SeqUtils import seq1

__all__ = ["calc_dihedral", "calc_chi1", "softmin_sample", "softmax_sample"]

# the 4 atoms that define the CA-CB rotational angle
CHI1_ATOMS = dict(ALA=None, GLY=None,
                 ARG=['N', 'CA', 'CB', 'CG'], ASN=['N', 'CA', 'CB', 'CG'],
                 ASP=['N', 'CA', 'CB', 'CG'], CYS=['N', 'CA', 'CB', 'SG'],
                 GLN=['N', 'CA', 'CB', 'CG'], GLU=['N', 'CA', 'CB', 'CG'],
                 HIS=['N', 'CA', 'CB', 'CG'], ILE=['N', 'CA', 'CB', 'CG1'],
                 LEU=['N', 'CA', 'CB', 'CG'], LYS=['N', 'CA', 'CB', 'CG'],
                 MET=['N', 'CA', 'CB', 'CG'], PHE=['N', 'CA', 'CB', 'CG'],
                 PRO=['N', 'CA', 'CB', 'CG'], SER=['N', 'CA', 'CB', 'OG'],
                 THR=['N', 'CA', 'CB', 'OG1'], TRP=['N', 'CA', 'CB', 'CG'],
                 TYR=['N', 'CA', 'CB', 'CG'], VAL=['N', 'CA', 'CB', 'CG1'])
"""
Atoms required to calculate chiral dihedral angles from each amino acid
"""

def calc_dihedral(p):
    #pylint: disable=invalid-name
    """
    Calculate dihedral angles between 4 cartesian points.

    Calculate dihedral angles between 4 cartesian points, meaning the angle between the plane defined by ABC and that defined by BCD from the four points passed (ABCD).
    The points should be on the first axis of the numpy array (i.e. in rows with coordinates as columns as displayed).
    The Phi, Psi and Omega backbone angles in proteins are dihedral angles between the planes defined by different combinations of backbone atoms (Phi: CA-C-N-CA, Psi: C-N-CA-C, Omega: N-CA-C-N).

    Parameters
    ----------
    p     : ndarray
        Numpy array of points. Different points are on the first axis with coordinates along the second axis. The points are ordered ABCD.

    Returns
    -------
    Float
        The calculated dihedral angle.

    References
    ----------
    This code is adapted from user Praxeolitic's `StackOverflow answer <https://stackoverflow.com/a/34245697>`_.
    """
    b0 = p[0] - p[1]
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def calc_chi1(mmcif, target_chain, seq):
    """
    Calculate Chi1 angle for each residue in an mmCIF dictionary.

    Calculate side chain rotamer chiral angle for each residue in an mmCIF dictionary.
    This angle is the dihedral angle (see `calc_dihedral`) between the backbone and side chain (specific atoms used can be found in `CHI1_ATOMS`).

    Parameters
    ----------
    mmcif        : dict
        mmCIF dictionary, as generated by Biopythons Bio.PDB.MMCIF2Dict.
    target_chain : str
        Chain to calculate angles for.
    seq          : Iterable of str
        Expected target chain sequence.

    Returns
    -------
    ndarray
        The calculated Chi1 angles for each position in the sequence.

    References
    ----------
    This function is adapted from `code by Umberto Perron <https://bitbucket.org/uperron/proteinnet_vep/src/master/continuous_angle_features.py>`_.

    """
    chain = np.array(mmcif.get("_atom_site.label_asym_id"))
    group = np.array(mmcif.get("_atom_site.group_PDB"))
    residue_name = np.array(mmcif.get("_atom_site.label_comp_id"))

    # Filter to correct atoms
    ind = ((chain == target_chain) &\
           (group == 'ATOM') &\
           (np.isin(residue_name, list(CHI1_ATOMS.keys()))))

    residue_name = residue_name[ind]
    residue_number = np.array(mmcif.get("_atom_site.label_seq_id"))[ind].astype(int)
    atom = np.array(mmcif.get("_atom_site.label_atom_id"))[ind]
    coords = np.vstack([np.array(mmcif.get("_atom_site.Cartn_x"), dtype=float)[ind],
                        np.array(mmcif.get("_atom_site.Cartn_y"), dtype=float)[ind],
                        np.array(mmcif.get("_atom_site.Cartn_z"), dtype=float)[ind]]
                      ).T

    chi = []
    mask = []
    for i, pn_residue in enumerate(seq, 1):
        # Not all residues have structure
        if i not in residue_number:
            chi.append(0)
            mask.append(0)
            continue

        # Ala and Gly don't have Chi1
        if pn_residue in ['A', 'G']:
            chi.append(0)
            mask.append(1)
            continue

        # Select correct atom coords
        # mmCIF dict automatically generates these with atoms correctly ordered for this opperation
        ind = residue_number == i
        res_name = residue_name[ind][0]
        res_atom = atom[ind]
        res_coords = coords[ind][np.isin(res_atom, CHI1_ATOMS[res_name])]

        if not seq1(res_name) == pn_residue:
            raise ValueError(f'PDB seq does not match ProteinNet seq at position {i}')

        # Don't have correct data
        if not res_coords.shape == (4, 3):
            chi.append(0)
            mask.append(0)
            continue

        chi.append(calc_dihedral(res_coords))
        mask.append(1)

    return np.array(chi, dtype=np.float), np.array(mask, dtype=np.int)

def add_neighbours(mask, max_value=np.infty):
    """
    Add neighbouring numbers to a 1D array

    Expand an array with all positive number 1 bigger or smaller than current entries, if not already present, and sort results.
    This will work for decimal values but generally only makes sense to apply to integers.

    Parameters
    ----------
    mask      : ndarray
        Array to expand with neighbours.
    max_value :
        Maximum value to include.

    Returns
    -------
    ndarray
        Sorted array containing all numbers from the original `mask` and values +/- from them.
    """
    mask = np.concatenate((mask - 1, mask, mask + 1))
    mask = mask[np.logical_and(mask >= 0, mask <= max_value)]
    return np.unique(mask)

def softmin_sample(weights, max_value=float('inf')):
    """
    Sample from weights after appling the softmin function to them, subject to the weight being less than `max_value`.

    Sample from weights after appling the softmin function to them, subject to the weight being less than `max_value`.
    Weights greater than `max_value` are removed prior to the softmin opperation.
    The output index corresponding to the index of the original array, with any values greater than `max_value` included.

    Parameters
    ----------
    weights   : ndarray
        Weights to sample from.
    max_value : float
        Maximum weight to consider

    Returns
    -------
    int
        Index selected by softmin weighted sampling.
    """
    inds = np.nonzero(weights < max_value)[0]
    weights = weights[inds]
    weights = np.exp(-weights)
    weights = weights / np.sum(weights)

    return np.random.choice(inds, p=weights)

def softmax_sample(weights, min_value=float('-inf')):
    """
    Sample from weights after appling the softmax function to them, subject to the weight being greater than `min_value`.

    Sample from weights after appling the softmax function to them, subject to the weight being greater than `min_value`.
    Weights less than `min_value` are removed prior to the softmax opperation.
    The output index corresponding to the index of the original array, with any values less than `min_value` included.

    Parameters
    ----------
    weights   : ndarray
        Weights to sample from.
    min_value : float
        Minimum weight to consider

    Returns
    -------
    int
        Index selected by softmax weighted sampling.
    """
    inds = np.nonzero(weights > min_value)[0]
    weights = weights[inds]
    weights = np.exp(weights)
    weights = weights / np.sum(weights)

    return np.random.choice(inds, p=weights)
