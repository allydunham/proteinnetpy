#!/usr/bin/env python3
"""
Filter a ProteinNet file to include/exclude specific records
"""
import argparse
import sys

from proteinnetpy.data import ProteinNetDataset

def main():
    """
    Filter a ProteinNet file to only include records from a list of IDs
    """
    args = arg_parser().parse_args()

    id_list = args.ids
    if args.file:
        if args.file == "-":
            id_list.extend([i.strip() for i in sys.stdin])
        else:
            with open(args.file, "r") as id_file:
                id_list.extend([i.strip() for i in id_file])

    if args.exclude:
        def filter_func(record): return record.id not in id_list
    else:
        def filter_func(record): return record.id in id_list

    data = ProteinNetDataset(path=args.proteinnet, preload=False, filter_func=filter_func)

    for rec in data:
        print(rec, file=sys.stdout)

def arg_parser():
    """Process arguments"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('proteinnet', metavar='P', help="Input ProteinNet file")
    parser.add_argument('ids', metavar='I', nargs="*", help="IDs list")

    parser.add_argument('--file', '-f', type=argparse.FileType("r"),
                        help="File containing one ID per line")
    parser.add_argument('--exclude', '-e', action="store_true", help="Exclude IDs")


    return parser

if __name__ == "__main__":
    main()