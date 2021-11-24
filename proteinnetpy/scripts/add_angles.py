"""
Add backbone and dihedral angles to a text ProteinNet data file.

Angles are added under new identifiers [RAMA] and [CHI], each with a final mask row.
Backbone angles are in rows ordered omega, phi, psi.
Dihedral angles require a local PDB database to query from.
"""
import sys
import os
import argparse
import gzip

import numpy as np
from Bio.PDB import MMCIF2Dict

from proteinnetpy.parser import record_parser
from proteinnetpy.maths import calc_chi1

class PDBeList:
    """
    Simple parser for local PDB(e) databases, in the style of Bio.PDB.PDBList
    """
    dir_formats = {'pdb': 'pdb', 'mmcif': 'mmCIF'}
    def __init__(self, root, gzip=False, default_format='mmCIF'):
        if not os.path.isdir(root):
            raise FileNotFoundError('Directory {root} does not exist')

        self.file_formats = {'pdb': 'pdb{}.ent', 'mmcif': '{}.cif'}
        if gzip:
            self.file_formats = {k: v + '.gz' for k, v in self.file_formats.items()}

        self.default_format = default_format

        root = root.rstrip('/')
        self.root = root

        # Check avilability of different options
        self._all = os.path.isdir(f'{root}/all')
        self._divided = os.path.isdir(f'{root}/divided')

    def fetch_path(self, pdb_id, file_format=None):
        """
        Retrieve a file path for a given entry from the database, checking it exists.
        """
        if file_format is None:
            file_format = self.default_format

        pdb_id = pdb_id.lower()

        file = self.file_formats[file_format.lower()].format(pdb_id)
        folder = self.dir_formats[file_format.lower()]

        # Try divided first
        if self._divided:
            path = f'{self.root}/divided/{folder}/{pdb_id[1:3]}/{file}'
            if os.path.isfile(path):
                return path

        # next try all
        if self._all:
            path = f'{self.root}/all/{folder}/{file}'
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(f'No {file_format} file found')

def chi1_missing(record, reason, skip_missing):
    """
    Handle an inability to add Chi1 angles to a record
    """
    if skip_missing:
        print(f'{record.id}: Skipped - {reason}', file=sys.stderr)
    else:
        print(f'{record.id}: Masking CHI1 angles - {reason}', file=sys.stderr)
        record.chi = np.zeros(len(record))
        record.chi_mask = np.zeros(len(record))

def main():
    """Main script"""
    args = parse_args()

    pdb_db = PDBeList(args.pdb, gzip=args.gzip, default_format='mmCIF') if args.chi else None
    opener = gzip.open if args.gzip else open

    for record in record_parser(args.pn_file):
        if args.chi:
            if not record.source == 'pdb':
                chi1_missing(record, f'source = {record.source}', args.filter)
                if args.filter:
                    continue

            else:
                # Calculate Chi angles from PDB
                try:
                    pdb_path = pdb_db.fetch_path(record.pdb_id)
                    with opener(pdb_path, 'rt') as mmcif_file:
                        mmcif = MMCIF2Dict.MMCIF2Dict(mmcif_file)
                        record.chi, record.chi_mask = calc_chi1(mmcif,
                                                                record.pdb_chain,
                                                                record.primary)

                except FileNotFoundError:
                    chi1_missing(record, 'no mmCIF file', args.filter)
                    if args.filter:
                        continue

                except ValueError as err:
                    chi1_missing(record, str(err), args.filter)
                    if args.filter:
                        continue

        if args.rama:
            record.calculate_backbone_angles()

        print(record)

def arg_parser():
    """Argument parser"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pn_file', metavar='P', help="Input")

    parser.add_argument('--pdb', '-p', default='data/pdb',
                        help="Path to root of local PDB directory, expects mmCIF to be available")

    parser.add_argument('--gzip', '-g', action='store_true',
                        help="Local PDB files are gzipped")

    parser.add_argument('--filter', '-f', action='store_true',
                        help="Dont print records where no structure is found")

    parser.add_argument('--rama', '-r', action='store_true',
                        help="Calculate ramachandran angles")

    parser.add_argument('--chi', '-c', action='store_true',
                        help="Calculate chi1 angles")

    return parser

def parse_args():
    """Process input arguments"""
    args = arg_parser().parse_args()

    if not args.rama and not args.chi:
        raise ValueError('Set either --rama or --chi, otherwise nothing is added')

    return args

if __name__ == "__main__":
    main()