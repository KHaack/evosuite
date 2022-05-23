# -*- coding: utf-8 -*-
"""
    Purpose: Compares the experiment results from two run of the the experiment_runner
    Author: Kevin Haack
"""
import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_CSV_A = "results\\2022-05-20 BR0.2\\default parameter"
DIRECTORY_CSV_B = "results\\2022-05-20 BR0.2\\0.85 parameter"
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# filter
FILTER_MIN_EXECUTIONS = 3


def getStatisticFiles(path):
    """
    Determines all statistic files.
    Returns: A list of all files
    """
    logging.debug('use pattern: ' + path);
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
    dataframe = ex.addAdditionalColumns(dataframe)
    dataframe = ex.getMeasurings(dataframe, 0)
    dataframe = ex.filter(dataframe, FILTER_MIN_EXECUTIONS)

    logging.info("filter...")
    percentReached = 2

    subset = ex.getMeasurings(dataframe, percentReached)
    subset = subset[subset['PercentageReached'] > 0]
    subset = subset[subset['_GradientBra'].eq(0)]

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

    logging.info("plot...")

    ax = plt.axes(projection='3d')
    ax.scatter3D(merged['_BranchRatio_b'], merged['_Fitness_b'], merged['_CoverageDifference'])
    ax.set_xlabel('_BranchRatio_b at 20%')
    ax.set_ylabel('_Fitness_b at 20%')
    ax.set_zlabel('_CoverageDifference')
    plt.show()


def plot(name, dataframe, x, y, save=False):
    """
    Plot the passed variables x and y.

    :param name: The name of that dataframe.
    :param dataframe: The dataframe.
    :param x: Variable x.
    :param y: Variable y.
    :param save: Indicates whether the plot should be save as file.
    :return: None
    """
    plotDir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_PLOT)

    if not os.path.exists(plotDir):
        os.mkdir(plotDir)

    dataframe.plot.scatter(x=x, y=y, title=name)

    if save:
        filename = name + "-" + x + "-" + y
        plotFile = os.path.join(plotDir, filename + ".png")
        plt.savefig(plotFile)
        plt.close(plotFile)


def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    logging.info("search files...")
    dataframeA = getStatisticFiles(os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CSV_A, '*.csv'))
    dataframeB = getStatisticFiles(os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CSV_B, '*.csv'))

    dataframeA['set'] = 'A'
    dataframeB['set'] = 'B'

    logging.info("start evaluation...")
    compare(pd.concat([dataframeA, dataframeB], ignore_index=True))


if __name__ == "__main__":
    main()
