"""
Record class to represent ProteinNet data entries
"""
import logging
import re
import numpy as np
from .maths import calc_dihedral, add_neighbours

# Regexs for recognising different ProteinNet file ID patterns
PDB_REGEX = re.compile(r"^[0-9A-Z]{4}_[0-9]*_[A-Z0-9]*$")
ASTRAL_REGEX = re.compile(r"^[0-9A-Z]{4}_[a-z0-9\-]{7}$")
CASP_REGEX = re.compile(r"^T[0-9]{4}$")

# Alphabetical cannonical encoding
AMINO_ACIDS = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                        'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
NUM_AAS = len(AMINO_ACIDS)
AA_HASH = {aa: index for index, aa in enumerate(AMINO_ACIDS)}

class ProteinNetRecord:
    # pylint: disable=invalid-name
    # pylint: disable=redefined-builtin
    """
    A record from a ProteinNet database, with support for torsion angles and additional
    per-position profiles (e.g. precalculated from something like UniRep)
    """
    def __init__(self, id, primary, evolutionary=None, info_content=None, secondary=None,
                 tertiary=None, mask=None, rama=None, rama_mask=None, chi=None,
                 chi_mask=None, profiles=None):
        self.id = id

        # Manage the different properties for Testing/Validation/Training sets
        self.split = None
        self.record_class = None
        self.source = None
        self.casp_id = None
        self.astral_id = None
        self.pdb_id = None
        self.pdb_chain_number = None
        self.pdb_chain = None

        if '#' in id:
            id = id.split('#')
            self.record_class = id[0]
            self.split = 'validation' # switched in _get_id_properties if should be test
            self._set_id_properties(id[1])
        else:
            self.split = 'training'
            self._set_id_properties(id)

        self.evolutionary = evolutionary
        self.info_content = info_content

        self.primary = primary
        self.primary_ind = self.enumerate_sequence() # precompute numeric sequence representation

        self.secondary = secondary

        self.tertiary = tertiary
        self.mask = mask

        self.rama = rama
        self.rama_mask = rama_mask

        self.chi = chi
        self.chi_mask = chi_mask

        self.profiles = profiles

        self._normalised_angles = False

        self._validate_masks()

    def _set_id_properties(self, id):
        """
        Extract details from the identifier
        """
        if CASP_REGEX.match(id):
            self.split = 'testing'
            self.casp_id = id
            self.source = 'casp'

        elif ASTRAL_REGEX.match(id):
            self.astral_id = id
            self.source = 'astral'

        elif PDB_REGEX.match(id):
            pdb = id.split('_')
            self.pdb_id = pdb[0]
            self.pdb_chain_number = pdb[1]
            self.pdb_chain = pdb[2]
            self.source = 'pdb'

    def _validate_masks(self):
        """
        Check masks exist for all fileds that expect them and generate permissive masks
        where they are missing, with a warning.
        """
        warn_msg = "'%s' is set without '%s', generating a permisive version"

        # evolutationary and info content
        if self.evolutionary is not None and self.info_content is None:
            logging.warning(warn_msg, 'evolutionary', 'info_content')
            self.info_content = np.ones(len(self.primary))

        # tertiary and mask
        if self.tertiary is not None and self.mask is None:
            logging.warning(warn_msg, 'tertiary', 'mask')
            self.mask = np.ones(len(self.primary))

        # rama and rama_mask
        if self.rama is not None and self.rama_mask is None:
            logging.warning(warn_msg, 'rama', 'rama_mask')
            self.rama_mask = np.ones(len(self.primary))

        # chi and chi_mask
        if self.chi is not None and self.chi_mask is None:
            logging.warning(warn_msg, 'chi', 'chi_mask')
            self.chi_mask = np.ones(len(self.primary))

    def __len__(self):
        return len(self.primary)

    def __str__(self):
        self._validate_masks()
        output = ['[ID]', self.id, '[PRIMARY]', ''.join(self.primary)]

        if self.evolutionary is not None:
            output.append('[EVOLUTIONARY]')
            output.extend('\t'.join(np.format_float_positional(i) for i in aa) for
                          aa in self.evolutionary)
            output.append('\t'.join(np.format_float_positional(i) for i in self.info_content))

        if self.secondary is not None:
            # output.append('[SECONDARY]')
            # Secondary not implemented yet
            pass

        if self.tertiary is not None:
            if self.mask is None:
                raise ValueError('tertiary set without mask')

            output.append('[TERTIARY]')
            # tertiary_reshaped = self.tertiary.reshape((len(self)*3, 3)).T.astype(str)
            output.extend('\t'.join(np.format_float_positional(i) for i in coord) for
                          coord in self.tertiary)

            output.append('[MASK]')
            output.append(''.join('+' if x == 1 else '-' for x in self.mask))

        if self.rama is not None:
            output.append('[RAMA]')
            output.extend('\t'.join(np.format_float_positional(i) for i in angle) for
                          angle in self.rama)
            output.append(''.join('+' if x == 1 else '-' for x in self.rama_mask))

        if self.chi is not None:
            output.append('[CHI]')
            output.append('\t'.join(np.format_float_positional(i) for i in self.chi))
            output.append(''.join('+' if x == 1 else '-' for x in self.chi_mask))

        return '\n'.join(output) + '\n'

    def enumerate_sequence(self, aa_hash=AA_HASH):
        """
        Generate a numeric representation of the sequence with each
        amino acid represented by an interger index, as mapped in aa_hash.
        """
        return np.array([aa_hash[aa] for aa in self.primary], dtype=np.int)

    def get_one_hot_sequence(self, aa_hash=AA_HASH):
        """
        Generate a 1-hot encoded matrix of the proteins sequence.
        aa_hash maps amino acids to their row number.
        """
        num_aas = max(v for k, v in aa_hash.items()) + 1
        indeces = np.array([aa_hash[aa] for aa in self.primary])
        one_hot = np.zeros((num_aas, len(indeces)), dtype=np.int)
        one_hot[indeces, np.arange(len(indeces))] = 1

        return one_hot

    def calculate_backbone_angles(self):
        """
        Calculate Omega, Phi, and Psi from the ProteinNet 3D coordinates and set them
        as instance variables
        """
        coords = self.tertiary.T
        psi = np.array([calc_dihedral(coords[i:(i+4)]) for i in range(0, 3*len(self)-3, 3)] + [0])
        phi = np.array([calc_dihedral(coords[i:(i+4)]) for i in range(2, 3*len(self)-3, 3)] + [0])
        omega = np.array([0] + [calc_dihedral(coords[i:(i+4)]) for i
                                in range(1, 3*len(self)-3, 3)])

        # identify positions where the angle is undefined
        unknown_inds = add_neighbours(np.argwhere(self.mask == 0).flatten(),
                                      max_value=len(self) - 1)
        psi[unknown_inds] = 0
        phi[unknown_inds] = 0
        omega[unknown_inds] = 0

        self.rama = np.stack([omega, phi, psi])
        self.rama_mask = np.ones_like(phi)
        self.rama_mask[unknown_inds] = 0

    def normalise_angles(self, factor=np.pi):
        """
        Normalise backbone and chi angles to betweeen -1 and 1. Will only
        apply normalisation once per record.
        """
        if not self._normalised_angles:
            self._normalised_angles = True
            self.rama = self.rama / factor
            self.chi = self.chi / factor

    def distance_matrix(self):
        """
        Calculate the distance matrix between residues C-alpha atoms
        """
        calpha = np.copy(self.tertiary[:, 1::3])

        x = calpha[0, :]
        y = calpha[1, :]
        z = calpha[2, :]

        dx = x[..., np.newaxis] - x[np.newaxis, ...]
        dy = y[..., np.newaxis] - y[np.newaxis, ...]
        dz = z[..., np.newaxis] - z[np.newaxis, ...]

        return (np.array([dx, dy, dz]) ** 2).sum(axis=0) ** 0.5
