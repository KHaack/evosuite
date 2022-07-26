# -*- coding: utf-8 -*-
"""
    Purpose: Compares the experiment results from two run of the experiment_runner
    Author: Kevin Haack
"""
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# filter
FILTER_MIN_EXECUTIONS = 25
SCATTER_POINT_SIZE = 4


def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    argument_parser = argparse.ArgumentParser(
        description="Compares the experiment results from two runs (a & b) of the the experiment_runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument("-a", help="The directory path of the run a", type=ex.check_dir_path, required=True)
    argument_parser.add_argument("-b", help="The directory path of the run b", type=ex.check_dir_path, required=True)
    argument_parser.add_argument("-only_pc", help="compares only classes with pc=yes", action='store_true')
    argument_parser.add_argument("-export", help="export the classes", action='store_true')

    return argument_parser


def get_dataframes():
    """
    Returns the two results in one dataframe.
    :return: The dataframe
    """
    dataframe_a = ex.get_statistics(args.a)
    dataframe_b = ex.get_statistics(args.b)

    dataframe_a['set'] = 'A'
    dataframe_b['set'] = 'B'

    return pd.concat([dataframe_a, dataframe_b], ignore_index=True)


def compare(dataframe):
    logging.info("compare...")
    logging.info(f"coverage difference (sum):\t{dataframe['Coverage difference (b-a)'].sum()}")
    logging.info(f"coverage difference (mean):\t{dataframe['Coverage difference (b-a)'].mean()}")
    logging.info(f"coverage difference (median):\t{dataframe['Coverage difference (b-a)'].median()}")
    logging.info(f"coverage difference (std):\t{dataframe['Coverage difference (b-a)'].std()}")
    logging.info(f"coverage difference (min):\t{dataframe['Coverage difference (b-a)'].min()}")
    logging.info(f"coverage difference (max):\t{dataframe['Coverage difference (b-a)'].max()}")

    print(dataframe)


def figures(dataframe, merged):
    logging.info("plot...")

    # histogram coverage difference
    ax = merged.hist(column='Coverage difference (b-a)', bins=50)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Coverage difference (b-a)")
    plt.title('Histogram - Coverage difference ($\\alpha - \\beta$)')
    plt.tight_layout()
    plt.show()

    # histogram coverage compare
    b = dataframe[dataframe['set'].eq('B')]
    a = dataframe[dataframe['set'].eq('A') & dataframe['TARGET_CLASS'].isin(b['TARGET_CLASS'])]

    ax = a.hist(column='EndCoverage', bins=20, alpha=0.5, legend=True)
    ax = b.hist(column='EndCoverage', bins=20, alpha=0.5, ax=ax, legend=True)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Coverage")
    ax[0][0].legend(["$\\alpha$", "$\\beta$"])
    plt.title('Histogram - Coverage')
    plt.tight_layout()
    plt.show()


def main():
    """
    Runs large scale experiment.
    """
    dataframe = get_dataframes()

    logging.info("------------------------------------------")

    logging.info("Total tests in A:\t\t" + str(len(dataframe[dataframe['set'].eq('A')].index)))
    logging.info("Total tests in B:\t\t" + str(len(dataframe[dataframe['set'].eq('B')].index)))
    logging.info("Total tests in B with PC:\t" + str(
        len(dataframe[dataframe['set'].eq('B') & dataframe["_ParameterControlled"].eq("yes")].index)))

    if args.only_pc:
        dataframe = dataframe[
            dataframe['set'].eq('A') | (dataframe['set'].eq('B') & dataframe["_ParameterControlled"].eq("yes"))]

    dataframe = ex.clean(dataframe)
    dataframe = ex.add_additional_columns(dataframe)

    logging.info("start compare...")
    dataframe = ex.get_measurements(dataframe, 2)

    logging.info("merge...")
    groups = dataframe.groupby(['set', 'TARGET_CLASS']).agg({
        'EndCoverage': ['median', 'std'],
        'BranchRatio': 'median',
        'Generations': ['median', 'max']
    })

    groups.reset_index(inplace=True)
    groups.sort_index(inplace=True)

    a = groups[groups['set'].eq('A')]
    b = groups[groups['set'].eq('B')]

    logging.info("Java in A classes:\t\t" + str(len(a.index)))
    logging.info("Java in B classes:\t\t" + str(len(b.index)))

    merged = a.merge(b, how='inner', left_on='TARGET_CLASS', right_on='TARGET_CLASS', suffixes=('_a', '_b'))
    logging.info("matched Java classes:\t\t" + str(len(merged.index)))

    logging.info("Generations median (A):\t\t" + str(merged[('Generations_a', 'median')].median()))
    logging.info("Generations median (B):\t\t" + str(merged[('Generations_b', 'median')].median()))
    logging.info("Generations max (A):\t\t" + str(merged[('Generations_a', 'max')].max()))
    logging.info("Generations max (B):\t\t" + str(merged[('Generations_b', 'max')].max()))

    merged['Coverage difference (b-a)'] = merged[('EndCoverage_b', 'median')] - merged[('EndCoverage_a', 'median')]

    if args.export:
        group = merged.groupby(['TARGET_CLASS']).agg({
            ('EndCoverage_a', 'median'): 'median',
            ('EndCoverage_b', 'median'): 'median',
            ('Coverage difference (b-a)', ''): 'median'
        })
        group.to_csv('export.csv')

    compare(merged)
    figures(dataframe, merged)


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
