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
    argument_parser.add_argument("-a", help="The directory path of the run a", type=ex.dir_path, required=True)
    argument_parser.add_argument("-b", help="The directory path of the run b", type=ex.dir_path, required=True)

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


def main():
    """
    Runs large scale experiment.
    """
    dataframe = get_dataframes()

    logging.info("------------------------------------------")

    logging.info("Total tests in A:\t\t" + str(len(dataframe[dataframe['set'].eq('A')].index)))
    logging.info("Total tests in B:\t\t" + str(len(dataframe[dataframe['set'].eq('B')].index)))

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

    logging.info("Generations median (A):\t\t" + str(merged[('_Generations_a', 'median')].median()))
    logging.info("Generations median (B):\t\t" + str(merged[('_Generations_b', 'median')].median()))
    logging.info("Generations max (A):\t\t" + str(merged[('_Generations_a', 'max')].max()))
    logging.info("Generations max (B):\t\t" + str(merged[('_Generations_b', 'max')].max()))

    merged['Coverage difference (a-b)'] = merged[('Coverage_a', 'median')] - merged[('Coverage_b', 'median')]

    logging.info(f"coverage difference (sum):\t{merged['Coverage difference (a-b)'].sum()}")
    logging.info(f"coverage difference (mean):\t{merged['Coverage difference (a-b)'].mean()}")
    logging.info(f"coverage difference (median):\t{merged['Coverage difference (a-b)'].median()}")
    logging.info(f"coverage difference (std):\t{merged['Coverage difference (a-b)'].std()}")
    logging.info(f"coverage difference (min):\t{merged['Coverage difference (a-b)'].min()}")
    logging.info(f"coverage difference (max):\t{merged['Coverage difference (a-b)'].max()}")

    logging.info("plot...")

    # histogram
    merged.hist(column='Coverage difference (a-b)', bins=50)
    plt.show()

    # difference
    fig, ax = plt.subplots()
    ax.scatter(merged[('Coverage_a', 'std')], merged['Coverage difference (a-b)'], s=SCATTER_POINT_SIZE)

    ax.set_xlabel('Coverage A (std) at 20%')
    ax.set_ylabel('Coverage difference (a-b)')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_title(f'BranchRatio - Coverage difference (a-b)')

    plt.show()


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
