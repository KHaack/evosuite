# -*- coding: utf-8 -*-
"""
    Purpose: Collect the experiment results from the experiment_runner
    Author: Kevin Haack
"""
import glob
import logging
import os
import re
import numpy as np
import pandas as pd
import sys
import scipy.stats as stats

# files and folders
DIRECTORY_CORPUS = "SF110-20130704"
DIRECTORY_REPORTS = "reports"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.DEBUG
LOG_STREAM = sys.stdout

def getStatisticFiles():
    """
    Determines all statistic files.
    Returns: A list of all files
    """
    pattern = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, '*', '*', DIRECTORY_REPORTS, '*', 'statistics.csv')
    logging.debug('use pattern: ' + pattern);
    return list(glob.iglob(pattern, recursive=True))

def evaluation(dataframe):
    """
    Creates a evaluation for the passed dataframe.
    :param dataframe: The dataframe for the evaluation
    """
    dataframe.drop('NeutralityVolumeSequence', axis=1, inplace=True)
    dataframe.drop('NeutralityVolumeEpsilon', axis=1, inplace=True)
    dataframe.drop('criterion', axis=1, inplace=True)
    dataframe.drop('FitnessMax', axis=1, inplace=True)
    dataframe.drop('FitnessMin', axis=1, inplace=True)
    dataframe.drop('Total_Goals', axis=1, inplace=True)
    dataframe.drop('Covered_Goals', axis=1, inplace=True)
    dataframe.drop('BranchCoverage', axis=1, inplace=True)

    dataframe['Generations'] = dataframe['Generations'] + 1
    dataframe['NV/Gen'] = dataframe['NeutralityVolume'] / dataframe['Generations']

    print(dataframe.head(50))

    logging.info("Generations max: " + str(dataframe['Generations'].max()))
    logging.info("Java classes: " + str(len(dataframe.groupby('TARGET_CLASS'))))

    logging.info("----------------------------------------------------------")

    getCorrelation('all', dataframe)
    getCorrelation('without zero generations', dataframe[dataframe['Generations'].gt(1)])
    getCorrelation('less then 100% coverage', dataframe[dataframe['Coverage'].lt(1)])
    getCorrelation('less then 75% coverage', dataframe[dataframe['Coverage'].lt(0.75)])
    getCorrelation('less then 50% coverage', dataframe[dataframe['Coverage'].lt(0.5)])


def getCorrelation(name, dataframe):
    """
    Calculate the correlation of the passed dataframe.
    :param name: the name of the passed data.
    :param dataframe: the dataframe.
    """
    logging.info("----------------------------------------------------------")
    logging.info(name + " (count: " + str(len(dataframe)) + ")")

    # InformationContent
    logging.info("Pearson correlation and p-value: Coverage-InformationContent")
    logging.info(stats.pearsonr(dataframe['Coverage'], dataframe['InformationContent']))

    # NeutralityVolume
    logging.info("Pearson correlation and p-value: Coverage-NeutralityVolume")
    logging.info(stats.pearsonr(dataframe['Coverage'], dataframe['NeutralityVolume']))

    # NV/Gen
    logging.info("Pearson correlation and p-value: Coverage-NV/Gen")
    logging.info(stats.pearsonr(dataframe['Coverage'], dataframe['NV/Gen']))

def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    files = getStatisticFiles()
    logging.info("search statistic files...")
    logging.info("found statistic files: " + str(len(files)))

    li = []
    for file in files:
        li.append(pd.read_csv(file, delimiter=','))

    dataframe = pd.concat(li, axis=0, ignore_index=True)

    logging.info("start evaluation...")
    evaluation(dataframe)


if __name__ == "__main__":
    main()
