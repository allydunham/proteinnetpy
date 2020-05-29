#!/usr/bin/env python3
"""
Module containing maths functionss for working with ProteinNet datasets
"""
import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1

# TODO refactor pandas out as it is the only module its used in?

# the 4 atoms that define the CA-CB rotational angle
CHI1_DICT = dict(ALA=None, GLY=None,
                 ARG=['N', 'CA', 'CB', 'CG'], ASN=['N', 'CA', 'CB', 'CG'],
                 ASP=['N', 'CA', 'CB', 'CG'], CYS=['N', 'CA', 'CB', 'SG'],
                 GLN=['N', 'CA', 'CB', 'CG'], GLU=['N', 'CA', 'CB', 'CG'],
                 HIS=['N', 'CA', 'CB', 'CG'], ILE=['N', 'CA', 'CB', 'CG1'],
                 LEU=['N', 'CA', 'CB', 'CG'], LYS=['N', 'CA', 'CB', 'CG'],
                 MET=['N', 'CA', 'CB', 'CG'], PHE=['N', 'CA', 'CB', 'CG'],
                 PRO=['N', 'CA', 'CB', 'CG'], SER=['N', 'CA', 'CB', 'OG'],
                 THR=['N', 'CA', 'CB', 'OG1'], TRP=['N', 'CA', 'CB', 'CG'],
                 TYR=['N', 'CA', 'CB', 'CG'], VAL=['N', 'CA', 'CB', 'CG1'])

def calc_dihedral(p):
    #pylint: disable=invalid-name
    """
    Calculate dihedral angles between 4 cartesian points, the different points are on the
    first axis (rows as displayed).
    From user Praxeolitic's answer https://stackoverflow.com/a/34245697
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

def calc_chi1(mmcif, chain, seq):
    """
    Calculate chi angles from an mmCIF file.
    mmcif: dict output of Bio.PDB.MMCIF2Dict
    Adapted from Umberto's code:
    https://bitbucket.org/uperron/proteinnet_vep/src/master/continuous_angle_features.py
    """
    atoms = {'group' : mmcif.get("_atom_site.group_PDB"),
             'authour_chain' : mmcif.get("_atom_site.label_asym_id"),
             'authour_residue_number' : mmcif.get("_atom_site.auth_seq_id"),
             'chain' : mmcif.get("_atom_site.label_asym_id"),
             'residue_number' : mmcif.get("_atom_site.label_seq_id"),
             'residue_name' : mmcif.get("_atom_site.label_comp_id"),
             'atom' : mmcif.get("_atom_site.label_atom_id"),
             'x' :  mmcif.get("_atom_site.Cartn_x"),
             'y' : mmcif.get("_atom_site.Cartn_y"),
             'z' : mmcif.get("_atom_site.Cartn_z")}
    atoms = pd.DataFrame(atoms)
    atoms[['x', 'y', 'z']] = atoms[['x', 'y', 'z']].astype(float)

    # Filter to correct atoms
    atoms = atoms[(atoms['chain'].isin(chain)) &\
                      (atoms['group'] == 'ATOM') &\
                      (atoms['residue_name'].isin(CHI1_DICT.keys()))]

    atoms[['authour_residue_number',
           'residue_number']] = atoms[['authour_residue_number', 'residue_number']].astype(int)

    residues = atoms.groupby('residue_number')

    chi = []
    mask = []
    for i, pn_residue in enumerate(seq, 1):
        # Not all residues have structure
        if i not in residues.groups.keys():
            chi.append(0)
            mask.append(0)
            continue

        # Ala and Gly don't have Chi1
        if pn_residue in ['A', 'G']:
            chi.append(0)
            mask.append(1)
            continue

        residue = residues.get_group(i)
        residue_name = residue['residue_name'].values[0]

        if not seq1(residue_name) == pn_residue:
            raise ValueError(f'PDB seq does not match ProteinNet seq at position {i}')

        # mmCIF dict automatically generates a df with atoms correctly ordered for this opperation
        coords = residue[residue['atom'].isin(CHI1_DICT[residue_name])][['x', 'y', 'z']].values

        # Don't have correct data
        if not coords.shape == (4, 3):
            chi.append(0)
            mask.append(0)
            continue

        chi.append(calc_dihedral(coords))
        mask.append(1)

    return np.array(chi, dtype=np.float), np.array(mask, dtype=np.int)

def add_neighbours(mask, max_value=np.infty):
    """
    Add all neighbouring numbers to a 1D array
    """
    mask = np.concatenate((mask - 1, mask, mask + 1))
    mask = mask[np.logical_and(mask > -1, mask <= max_value)]
    return np.unique(mask)

def softmin_sample(weights, max_value=float('inf')):
    """
    Sample an index from the input weights, subject to the weight being less than
    max_value. Weighting is via the softmin to prioritise the lowest weights
    """
    inds = np.nonzero(weights < max_value)[0]
    weights = weights[inds]
    weights = np.exp(-weights)
    weights = weights / np.sum(weights)

    return np.random.choice(inds, p=weights)

def softmax_sample(weights, min_value=float('-inf')):
    """
    Sample an index from the input weights, subject to the weight being greater than
    min_value. Weighting is via the softmax to prioritise the highest weights
    """
    inds = np.nonzero(weights > min_value)[0]
    weights = weights[inds]
    weights = np.exp(weights)
    weights = weights / np.sum(weights)

    return np.random.choice(inds, p=weights)
