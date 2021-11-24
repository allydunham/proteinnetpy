"""
Generate a fasta file from a ProteinNet text data file
"""
import argparse
import textwrap

from proteinnetpy.parser import record_parser

def main():
    """Main script"""
    args = arg_parser().parse_args()

    wrapper = textwrap.TextWrapper(width=60, )
    for record in record_parser(args.pn_file):
        print('>', record.id, sep='')
        print(*wrapper.wrap(''.join(record.primary)), sep='\n')

def arg_parser():
    """Process input arguments"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pn_file', metavar='P', help="Input ProteinNet text file")

    return parser

if __name__ == "__main__":
    main()