"""
Parse ProteinNet files
"""
import logging
import numpy as np
from .record import ProteinNetRecord

__all__ = ["record_parser", "fetch_record"]

# ProteinNet file constants
LINES_PER_RECORD = 33 # currently doesn't include secondary structure
PROTEINNET_FIELDS = ['id', 'primary', 'secondary', 'tertiary', 'evolutionary',
                     'info_content', 'mask', 'rama', 'rama_mask', 'chi',
                     'chi_mask']

def record_parser(path, max_len=float('inf'), excluded_fields=None, normalise_angles=False,
                  profiles=None):
    """
    Parse records from a ProteinNet file.

    Generator yielding records from a ProteinNet file in record.ProteinNetRecord form.
    It also allows some minimal manipulation of records before creating a data.ProteinNetDataset, although most filtering is assumed to be done at the dataset stage.
    In general, this function does not need to be used directly by end users, since the data.ProteinNetDataset class manages loading from files and allows keywords to be passed to the parser.

    Parameters
    ----------
    path              : str
        Path to ProteinNet file
    max_len           : int
        Max length of sequences to keep
    excluded_fields   : list
        Fields to exclude from the output record, reducing memory usage if only specific data is required. You cannot exclude id or primary.
    normalise_angles  : bool
        Normalise backbone/chiral angles to be -1 to 1 rather than -pi to pi
    profiles          : function
        Function returning positional profiles for a given record or None if profiles are not available for that record. These profiles can be any additional data associated with positions in a protein, for example the output of a protein language model or surface accessibility data. Few expectations are placed on them in downstream code and equally they are rarely used.

    Yields
    ------
    ProteinNetRecord
        Records parsed and processed from the input ProteinNet file.
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
    Read a field from an open ProteinNet file and add it to rec_dict

    Read a ProteinNet field from an open ProteinNet file and add it to rec_dict.
    The ID line should contain a section header (e.g. [ID] or [PRIMARY]), which will be used to determine how to parse the following section.
    Nothing is returned, instead the data is added to `rec_dict`, which will build up into a dictionary with all data for the current record.
    This function is primarily used internally by record_parser.

    Parameters
    ----------
    id_line   : str
        ID line of the following section in the ProteinNet file.
    rec_dict  : dict
        Dictionary of data for the current record
    file      : file_handle
        Open ProteinNet file, from which the id_line has just been read.
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
    Retrieve a record from a ProteinNet file

    Fetch a specific record from a ProteinNet file by its ID.

    Parameters
    ----------
    record_id : str
        ID of the record to fetch.
    path      : str
        Path to a ProteinNet file to search in.

    Returns
    -------
    ProteinNetRecord or None
        Record with the given ID, if found, otherwise None.
    """
    for pn_record in record_parser(path):
        if pn_record.id == record_id:
            return pn_record

    return None
