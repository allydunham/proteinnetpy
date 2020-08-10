#!/usr/bin/env python3
"""
WARNING - Old script hasn't been updated with module at present
Analyse performance of current mutational model
"""
# TODO Update for new module
# TODO Move to Mutation module?
import os
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from . import parser as pp, generators as gens

# TODO rework to work with new module setup - maps etc.
# TODO estimate number of entries and make progress bar

def analyse_mutations(generator, output_dir, prefix='', max_sequences=None, file_size=None):
    """
    Analyse the type of mutations produced by generator

    generator - generator yielding test variant sequences
    """
    if max_sequences is None:
        max_sequences = float('inf')

    output_dir = output_dir.rstrip('/')

    mutations = mutant_generator_to_data_frame(generator,
                                               max_iterations=max_sequences,
                                               file_size=file_size)

    # Mutation Counts
    mutations['mut_count'] = mutations.positions.apply(len)
    fig, _ = mut_counts_barchart(mutations, bar_width=0.9)
    fig.savefig(f'{output_dir}/{prefix}_mutation_counts.pdf')

    # Proportional Position of mutations
    mutations['length'] = mutations.wt_seq.apply(len)
    mutations['proportional_position'] = mutations.apply(lambda x: (x['positions'] + 1)/x['length'],
                                                         axis=1)
    fig, _ = prop_pos_histogram(mutations)
    fig.savefig(f'{output_dir}/{prefix}_position_histogram.pdf')

    # Frequency of substitutions
    mut_fig, _, wt_fig, _ = plot_substitution_frequencies(mutations)
    mut_fig.savefig(f'{output_dir}/{prefix}_mutant_residues.pdf')
    wt_fig.savefig(f'{output_dir}/{prefix}_mutated_wt_residues.pdf')

# TODO add potential for max_iterations to be greater than file size?
def mutant_generator_to_data_frame(generator, max_iterations=float('inf'), file_size=None):
    """
    Generate mutations from a testing generator and parse them into a pd.DataFrame
    """
    if file_size is None:
        print('Warning: Unknown file size, progress bar may be inaccurate')
        size = max_iterations - 1 if max_iterations != float('inf') else None

    else:
        if max_iterations == float('inf'):
            size = file_size - 1
        elif max_iterations > file_size:
            print(f'Warning: max_iterations ({max_iterations}) > than file size {file_size}.',
                  'Currently only one run through the file is permitted')
            size = file_size - 1
        else:
            size = max_iterations - 1


    first_record = next(generator)
    data = {k:[] for k in first_record}
    _add_record_to_dict(data, first_record)

    i = 1
    for record in tqdm(generator, total=size):
        _add_record_to_dict(data, record)

        i += 1
        if i > max_iterations:
            break

    return pd.DataFrame(data)

def _add_record_to_dict(dic, rec):
    """
    Add a mutated ProteinNet record (in the dictionary format of test functions) to
    a dictionary containing lists of each feature, in the form used to create a pd.DataFrame
    """
    for key in ('wt_seq', 'mut_seq', 'substitutions'):
        rec[key] = pp.AMINO_ACIDS[rec[key]]

    for key, value in rec.items():
        dic[key].append(value)

def mut_counts_barchart(mutations, bar_width=0.9):
    """
    Plot counts of each mutation number (expecting a uniform distribution)
    """
    counts = mutations.mut_count.value_counts()

    fig, ax = plt.subplots()

    rects = ax.bar(counts.index, counts, bar_width, color='b')
    ax.set_xlabel('Number of Variants')
    ax.set_ylabel('Count')
    ax.set_title('Count of variants generated per protein')
    return fig, ax

def prop_pos_histogram(mutations, n_bins=10):
    """
    Plot a histogram of the proportional position of mutations within proteins
    """
    fig, axis = plt.subplots()
    axis.hist(np.concatenate(mutations.proportional_position), bins=n_bins, color='b')
    axis.set_xlabel('Position in protein')
    axis.set_ylabel('Count')
    axis.set_title('Position of generated variants in proteins')

    return fig, axis

def plot_substitution_frequencies(mutations, bar_width=0.9):
    """
    Plot counts of each AA substitution
    """
    # TODO Add AA colours

    originals = np.concatenate(mutations.apply(lambda x: x['wt_seq'][x['positions']], axis=1))
    substitutions = np.concatenate(mutations.substitutions)

    subs = pd.DataFrame({'wt': originals, 'mut': substitutions,
                         'sig': np.char.add(originals, np.char.add('>', substitutions))})

    mut_fig, mut_axs = signature_barchart(subs, primary='mut', bar_width=bar_width)
    wt_fig, wt_axs = signature_barchart(subs, primary='wt', bar_width=bar_width)

    return mut_fig, mut_axs, wt_fig, wt_axs

def signature_barchart(subs, primary='mut', bar_width=0.9):
    """
    Plot barchart showing occurance of mutations from/to (based on 'primary' = ['wt'/'mut']) each
    amino acid and which signatures are most common for that AA
    """
    fig, axs = plt.subplots(nrows=2, figsize=(40, 8), sharex=True)

    mut_counts = subs[primary].value_counts()
    mut_indeces = [pp.AA_HASH[aa] + 0.5 for aa in mut_counts.index]

    sig_counts = subs.groupby(primary).sig.value_counts().astype(np.float)
    for i in sig_counts.index.unique(level=primary):
        sig_counts[i] = sig_counts[i]/mut_counts[i]

    sig_index_hash = OrderedDict()
    if primary == 'mut':
        sig_hasher = lambda ind1, ind2: ind2 + 0.5 + bar_width * (ind1 - 9) / 20
    elif primary == 'wt':
        sig_hasher = lambda ind1, ind2: ind1 + 0.5 + bar_width * (ind2 - 9) / 20
    else:
        raise ValueError(f"'primary' must be one of [wt/mut], recieved {primary}")

    for ind1, aa1 in enumerate(pp.AMINO_ACIDS):
        for ind2, aa2 in enumerate(pp.AMINO_ACIDS):
            #if not aa1 == aa2: # keep out as sanity check - such subs should never happen
            sig = ''.join((aa1, '>', aa2))
            sig_index_hash[sig] = sig_hasher(ind1, ind2)

    sig_indeces = [sig_index_hash[sig] for sig in sig_counts.index.get_level_values('sig')]

    if primary == 'mut':
        title = 'Mutant AA count and per mutant WT AA frequency'
    else:
        title = 'WT AA count and per WT mutant AA frequency'

    axs[0].set_title(title, fontsize=16)
    axs[0].bar(mut_indeces, mut_counts, width=bar_width)
    axs[0].set_ylabel('Count')
    axs[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    axs[1].bar(sig_indeces, sig_counts, width=0.75*bar_width/20)
    axs[1].set_ylabel('Freq (per AA)')

    axs[1].set_xlabel('Mutation', fontsize='xx-large')
    axs[1].set_xlim(0, 20)
    axs[0].tick_params(axis='x', which='both', length=0)

    # Tick lines between each AA group
    axs[1].tick_params(axis='x', which='major', length=0, pad=25)
    axs[1].set_xticks(np.arange(0.5, 20.5, 1))
    axs[1].set_xticklabels(pp.AMINO_ACIDS, fontfamily='monospace', fontsize='xx-large')

    # Labels for each mutation signature
    axs[1].tick_params(axis='x', which='minor', length=0, pad=3)
    axs[1].set_xticks(list(sig_index_hash.values()), minor=True)
    axs[1].set_xticklabels(list(sig_index_hash.keys()), rotation=90,
                           minor=True, fontfamily='monospace', fontsize='x-small')

    # Large ticks between AA groups
    for i in np.arange(21):
        axs[1].axvline(x=i, ymax=0, ymin=-0.05, clip_on=False, color='k', linewidth=1)

    plt.tight_layout()
    return fig, axs

def main(args):
    """Main script"""
    os.makedirs(args.output_dir, exist_ok=True)

    # Try to fetch sizes for ProteinNet Files
    entries = None
    try:
        entries = int(args.file_size)
    except ValueError:
        with open(args.file_size, 'r') as file_size:
            casp, name = args.protein_net.split('/')[-2:]
            for i in file_size:
                # lines structured as tab separated casp name size
                i = i.split('\t')
                if i[0] == casp and i[1] == name:
                    entries = int(i[2])
                    break

    if entries is None:
        print(f'Warning: file size arg {args.file_sizes} is not an integer or a file containing',
              f'a size for chosen dataset ({args.protein_net}), proceeding with unknown size')

    del_generator = gens.pssm_multi_mut_seq_generator(args.protein_net, prop_negative=1,
                                                      max_mutations=args.max_mutations,
                                                      max_deleterious=args.max_deleterious,
                                                      min_neutral=args.min_neutral,
                                                      verbose=True)

    neut_generator = gens.pssm_multi_mut_seq_generator(args.protein_net, prop_negative=0,
                                                       max_mutations=args.max_mutations,
                                                       max_deleterious=args.max_deleterious,
                                                       min_neutral=args.min_neutral,
                                                       verbose=True)

    print('Analysing deleterious variants')
    analyse_mutations(del_generator, args.output_dir, prefix='deleterious',
                      max_sequences=args.number, file_size=entries)

    print('Analysing neutral variants')
    analyse_mutations(neut_generator, args.output_dir, prefix='neutral',
                      max_sequences=args.number, file_size=entries)

def _int_or_inf(x):
    """
    Allow infinity or integer for max number argument
    """
    if x == 'inf' or x is None:
        return(float('inf'))
    else:
        return(int(x))

def parse_args():
    """Process input arguments"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_dir', metavar='O', help='Directory to output plots and statistics')

    parser.add_argument('--protein_net', '-p',
                        default='data/protein_net/text/casp10/redux_training_30',
                        help='ProteinNet file to draw records from')

    parser.add_argument('--number', '-n', type=_int_or_inf, default='inf',
                        help="Number of proteins to generate mutations for")

    parser.add_argument('--max_mutations', '-m', type=int, default=5,
                        help="Maximum variants per protein")

    parser.add_argument('--max_deleterious', '-d', type=float, default=0.025,
                        help="Maximum PSSM value considered deleterious")

    parser.add_argument('--min_neutral', '-t', type=int, default=0.1,
                        help="Minimum PSSM value considered neutral")

    parser.add_argument('--file_size', '-f', default='meta/proteinnet_counts',
                        help="File containing ProteinNet file size (in records) or integer size")

    return parser.parse_args()

if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)