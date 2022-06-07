# -*- coding: utf-8 -*-
"""
    Purpose: Compares the experiment results from two run of the the experiment_runner
    Author: Kevin Haack
"""
import argparse
import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# filter
FILTER_MIN_EXECUTIONS = 9


def get_statistic_files(path):
    """
    Determines all statistic files.
    Returns: A list of all files
    """
    logging.debug('use pattern: ' + path)
    files = list(glob.iglob(path, recursive=True))

    li = []
    for file in files:
        li.append(pd.read_csv(file, delimiter=','))

    return pd.concat(li, axis=0, ignore_index=True)


def compare(dataframe):
    """
    Creates a evaluation for the passed dataframe.
    :param dataframe: The dataframe for the evaluation
    """
    a = dataframe[dataframe['set'].eq('A')]
    b = dataframe[dataframe['set'].eq('B')]

    logging.info("Total tests in A:\t" + str(len(a.index)))
    logging.info("Total tests in B:\t" + str(len(b.index)))

    dataframe = ex.clean(dataframe)
    dataframe = ex.add_additional_columns(dataframe)
    dataframe = ex.get_measurements(dataframe, 0)
    dataframe = ex.filter_dataframe(dataframe, FILTER_MIN_EXECUTIONS)

    logging.info("filter...")
    percent_reached = 2

    subset = ex.get_measurements(dataframe, percent_reached)

    logging.info("merge...")
    groups = subset.groupby(['set', 'TARGET_CLASS']).median()
    groups.reset_index(inplace=True)

    a = groups[groups['set'].eq('A')]
    b = groups[groups['set'].eq('B')]

    logging.info("Java in A classes:\t\t" + str(len(a.index)))
    logging.info("Java in B classes:\t\t" + str(len(b.index)))

    merged = a.merge(b, how='inner', left_on='TARGET_CLASS', right_on='TARGET_CLASS', suffixes=('_a', '_b'))
    merged['_CoverageDifference'] = merged['Coverage_a'] - merged['Coverage_b']

    logging.info(f"sum of coverage difference: {merged['_CoverageDifference'].sum()}")
    logging.info(f"mean coverage difference: {merged['_CoverageDifference'].mean()}")
    logging.info(f"median coverage difference: {merged['_CoverageDifference'].median()}")
    logging.info(f"std coverage difference: {merged['_CoverageDifference'].std()}")

    logging.info("plot...")

    ax = plt.axes(projection='3d')
    ax.scatter3D(merged['_BranchRatio_b'], merged['_Fitness_b'], merged['_CoverageDifference'])
    ax.set_xlabel('_BranchRatio_b at 20%')
    ax.set_ylabel('_Fitness_b at 20%')
    ax.set_zlabel('_CoverageDifference')
    plt.show()


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


def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    logging.info("search files...")
    dataframe_a = get_statistic_files(os.path.join(args.a, '**', '*.csv'))
    dataframe_b = get_statistic_files(os.path.join(args.b, '**', '*.csv'))

    dataframe_a['set'] = 'A'
    dataframe_b['set'] = 'B'

    logging.info("start evaluation...")
    compare(pd.concat([dataframe_a, dataframe_b], ignore_index=True))


if __name__ == "__main__":
    args = setup_argparse().parse_args()
    main()
