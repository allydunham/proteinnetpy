"""
Parse ProteinNet files
"""
import logging
import numpy as np
from .record import ProteinNetRecord

# ProteinNet file constants
LINES_PER_RECORD = 33 # currently doesn't include secondary structure
PROTEINNET_FIELDS = ['id', 'primary', 'secondary', 'tertiary', 'evolutionary',
                     'info_content', 'mask', 'rama', 'rama_mask', 'chi',
                     'chi_mask']

def record_parser(path, max_len=float('inf'), excluded_fields=None, normalise_angles=False,
                  profiles=None):
    """
    Generator yielding records from a ProteinNet file one by one.
    excluded_fields lets you specify a list of fields to exclude from the output.
    You cannot exclude id or primary.

    path: path to ProteinNet files
    max_len: max length of sequences to keep
    excluded_fields: list of fields to exclude
    normalise_angles: Normalise backbone/chiral angles to be -1 to 1 rather than -pi to pi
    profiles: function to supply profiles for a given record. Should return None if not available.
    """
    if excluded_fields is None:
        excluded_fields = []

    with open(path, 'r') as pn_file:
        record = {}
        while True:
            line = pn_file.readline()

            # Empty string means end of file
            if not line:
                break

            # Empty line means end of record
            elif line == '\n':
                # remove excluded fields
                record = {k: v for k, v in record.items() if
                          k in ['id', 'primary'] or not k in excluded_fields}

                pn_rec = ProteinNetRecord(**record)
                record = {}

                if normalise_angles:
                    pn_rec.normalise_angles()

                if profiles is not None:
                    pn_rec.profiles = profiles(pn_rec)

                if len(pn_rec) < max_len:
                    yield pn_rec
                continue

            # Header line marks start of field
            elif line[0] == '[':
                _read_proteinnet_field(id_line=line, rec_dict=record, file=pn_file)

            else:
                # Should never get here
                raise ValueError(('Line is not empty, a newline or a field header: file may '
                                  'not be formatted correctly or the parser is outdated'))

def _read_proteinnet_field(id_line, rec_dict, file):
    """
    Read a field from a ProteinNet file and add it to rec_dict
    """
    identifier = id_line.strip('\n[]').lower()
    if identifier == 'id':
        rec_dict['id'] = file.readline().strip()

    elif identifier == 'primary':
        rec_dict['primary'] = np.array(list(file.readline().strip()))

    elif identifier == 'evolutionary':
        mat = [file.readline().strip().split() for _ in range(20)]
        rec_dict['evolutionary'] = np.array(mat, dtype=np.float)
        rec_dict['info_content'] = np.array(file.readline().strip().split(),
                                            dtype=np.float)

    elif identifier == 'secondary':
        # Secondary struct not implemented in proteinnet yet
        pass

    elif identifier == 'tertiary':
        mat = [file.readline().strip().split() for _ in range(3)]
        rec_dict['tertiary'] = np.array(mat, dtype=np.float)

        # Can put in per atom form by
        # x = tertiary.T.reshape((len(mat[0]) // 3, 3, 3))
        # And convert back via x.reshape((len(x)*3, 3)).T.astype(str)

    elif identifier == 'mask':
        rec_dict['mask'] = np.array([1 if i == '+' else 0 for
                                     i in file.readline().strip()],
                                    dtype=np.int)

    elif identifier == 'rama':
        # three lines - omega, phi, psi
        mat = [file.readline().strip().split() for _ in range(3)]
        rec_dict['rama'] = np.array(mat, dtype=np.float)
        mask = np.array([1 if i == '+' else 0 for i in file.readline().strip()],
                        dtype=np.int)
        rec_dict['rama_mask'] = mask

    elif identifier == 'chi':
        rec_dict['chi'] = np.array(file.readline().strip().split(), dtype=np.float)
        mask = np.array([1 if i == '+' else 0 for i in file.readline().strip()],
                        dtype=np.int)
        rec_dict['chi_mask'] = mask

def fetch_record(record_id, path):
    """
    Retrieve a particular record from a ProteinNet file
    """
    for pn_record in record_parser(path):
        if pn_record.id == record_id:
            return pn_record

    return None

def load_unirep(record, root=None):
    """
    Load unirep profile for a ProteinNetRecord
    """
    # Profiles that we have have stop column and M to start if there wasn't one
    file_id = record.id
    if '#' in file_id: # validation IDs are of the form 30#PDB_ID
        file_id = file_id.split('#')[1]

    try:
        profs = np.load(f"{root}/{file_id}.fa.npy")[:-1,:]
    except FileNotFoundError as err:
        logging.warning('Record %s: file %s not found', record.id, err.filename)
        return None

    # Check if there's an appended Met
    if profs.shape[0] > len(record):
        profs = profs[1:,:]

    return profs

