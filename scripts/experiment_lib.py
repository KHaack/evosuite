"""
    Purpose: This experiment library contains shared functions for the experiments.
    Author: Kevin Haack
"""
import argparse
import logging
import os
import platform

import pandas as pd

# files and directories
FILE_CLASSES = "samples\\00 - original - 23894.txt"
FILE_EXPORT = "samples\\export.txt"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# filtering
FILTER_ZERO_GENERATIONS = True
FILTER_PERCENTAGE = True


def create_export(dataframe):
    """
    Export the passed dataframe to the runner format.
    :param dataframe: The dataframe to export
    :return: None
    """
    path_samples = os.path.join(PATH_WORKING_DIRECTORY, FILE_CLASSES)
    path_export = os.path.join(PATH_WORKING_DIRECTORY, FILE_EXPORT)

    samples = pd.read_csv(path_samples, delimiter='\t', names=['project', 'TARGET_CLASS'])
    merged = dataframe.merge(samples, left_on='TARGET_CLASS', right_on='TARGET_CLASS', how='inner')
    merged = merged.groupby(['project', 'TARGET_CLASS']).count().reset_index()[['project', 'TARGET_CLASS']]

    merged.to_csv(path_export, header=None, sep='\t', index=False)


def add_additional_columns(dataframe):
    dataframe['PercentageReached'] = dataframe['NeutralityVolume'].str.count(';') * 10

    # classes
    classes = {'worst': 0, 'bad': 1, 'mid': 2, 'good': 3, 'very good': 4}
    dataframe.loc[dataframe['Coverage'].le(0.20), 'CoverageClass'] = classes['worst']
    dataframe.loc[dataframe['Coverage'].gt(0.20) & dataframe['Coverage'].le(0.40), 'CoverageClass'] = classes['bad']
    dataframe.loc[dataframe['Coverage'].gt(0.40) & dataframe['Coverage'].lt(0.60), 'CoverageClass'] = classes['mid']
    dataframe.loc[dataframe['Coverage'].ge(0.60) & dataframe['Coverage'].lt(0.80), 'CoverageClass'] = classes['good']
    dataframe.loc[dataframe['Coverage'].ge(0.80), 'CoverageClass'] = classes['very good']

    # classification groundTruth
    dataframe.loc[dataframe['CoverageClass'].lt(4), 'GroundTruth'] = True
    dataframe.loc[dataframe['CoverageClass'].eq(4), 'GroundTruth'] = False

    # branchless
    dataframe['Branchless'] = dataframe['Total_Branches'].eq(0)

    return dataframe


def clean(dataframe):
    """
    Clean the passed dataframe.
    :param dataframe: The dataframe.
    :return: The cleaned dataframe.
    """
    # remove
    dataframe.drop('_FitnessMax', axis=1, inplace=True)
    dataframe.drop('_FitnessMin', axis=1, inplace=True)
    dataframe.drop('Fitness', axis=1, inplace=True)
    dataframe.drop('criterion', axis=1, inplace=True)
    dataframe.drop('Total_Goals', axis=1, inplace=True)
    dataframe.drop('Covered_Goals', axis=1, inplace=True)
    dataframe.drop('BranchCoverage', axis=1, inplace=True)
    dataframe.drop('Generations', axis=1, inplace=True)

    # rename
    dataframe.rename({
        '_InformationContent': 'InformationContent',
        '_NeutralityVolume': 'NeutralityVolume',
        '_FitnessRatio': 'Fitness',
        '_Generations': 'Generations',
        '_GradientBranchesCovered': 'GradientBranchesCovered',
        '_GradientBranches': 'GradientBranches'
    }, inplace=True, axis=1)

    return dataframe


def get_n_th(x, n):
    """
    Returns the n-th element of the passed string x in the format [a;b;c;d].
    :param x: String in the format [a;b;c;d]
    :param n: The n that should be returned.
    :return: The n-th element.
    """
    if x == '[]':
        return

    x = x.replace('[', '').replace(']', '')
    parts = x.split(';')

    if len(parts) <= n:
        return

    return float(parts[n])


def get_measurements(dataframe, percent):
    """
    Extract the measuring.
    :param dataframe: The dataframe.
    :param percent:
    :return:
    """
    # filter
    if percent >= 0:
        dataframe = dataframe[dataframe['PercentageReached'].ge(percent * 10)]

    index = percent - 1

    # get measurings
    dataframe['_GradientBra'] = dataframe['GradientBranches'].apply(lambda x: get_n_th(x, index))
    dataframe['_GradientCov'] = dataframe['GradientBranchesCovered'].apply(lambda x: get_n_th(x, index))
    dataframe['_Fitness'] = dataframe['Fitness'].apply(lambda x: get_n_th(x, index))
    dataframe['_InfoContent'] = dataframe['InformationContent'].apply(lambda x: get_n_th(x, index))
    dataframe['_NeutralityVol'] = dataframe['NeutralityVolume'].apply(lambda x: get_n_th(x, index))
    dataframe['_Generations'] = dataframe['Generations'].apply(lambda x: get_n_th(x, index))

    # calculate others
    dataframe.loc[dataframe['_Generations'].eq(0), '_NeutralityGen'] = 0
    dataframe.loc[dataframe['_Generations'].gt(0), '_NeutralityGen'] = dataframe['_NeutralityVol'] / (
        dataframe['_Generations'])
    dataframe.loc[dataframe['_NeutralityGen'] > 1, '_NeutralityGen'] = 1

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_NotGradRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_NotGradRatio'] = (dataframe['Total_Branches'] - dataframe[
        '_GradientBra']) / (dataframe['Total_Branches'])

    dataframe.loc[dataframe['_GradientBra'].eq(0), '_GradientRatio'] = 0
    dataframe.loc[dataframe['_GradientBra'].gt(0), '_GradientRatio'] = dataframe['_GradientCov'] / (
        dataframe['_GradientBra'])

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_BranchRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_BranchRatio'] = dataframe['_GradientBra'] / (
        dataframe['Total_Branches'])

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_NotCovGraRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_NotCovGraRatio'] = (dataframe['_GradientBra'] - dataframe[
        '_GradientCov']) / (dataframe['Total_Branches'])

    return dataframe


def filter_dataframe(dataframe, minimum_executions):
    """
    Filter the passed dataframe.
    :param minimum_executions: Datasets with lower executions will be filtered.
    :param dataframe: The dataframe to filter.
    :return: The filtered dataframe.
    """
    # not 10% reached
    if FILTER_PERCENTAGE:
        total_length = len(dataframe.index)
        dataframe = dataframe[dataframe['PercentageReached'] > 0]
        logging.info("Tests not reached 10%:\t" + str(total_length - len(dataframe.index)))

    # zero generations
    if FILTER_ZERO_GENERATIONS:
        total_length = len(dataframe.index)
        dataframe = dataframe[dataframe['_Generations'] > 0]
        logging.info("Zero generations tests:\t" + str(total_length - len(dataframe.index)))

    # executions
    if minimum_executions > 0:
        total_length = len(dataframe.index)
        groups = dataframe.groupby('TARGET_CLASS').count()
        groups = groups.reset_index()
        groups = groups[groups['Algorithm'] >= minimum_executions]
        dataframe = dataframe[dataframe['TARGET_CLASS'].isin(groups['TARGET_CLASS'])]
        logging.info(f"Tests less then {str(minimum_executions)}execs:\t{str(total_length - len(dataframe.index))}")

    return dataframe


def dir_path(path):
    """
    Retruns true, if the passed string is a directory path.
    :param path: the directory path to check.
    :return: Retruns true, if the passed string is a directory path.
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def file_path(path):
    """
    Retruns true, if the passed string is a file path.
    :param path: the file path to check.
    :return: Retruns true, if the passed string is a file path.
    """
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def shutdown():
    """
    Shutdown the system.
    :return: None
    """
    logging.info("shutdown...")
    system = platform.system()
    if system == 'Linux':
        os.system("sudo poweroff")
    elif system == 'Windows':
        os.system("shutdown /s /t 1")
    else:
        raise Exception(f"Unsupported os '{system}'")


def reboot():
    """
    Reboot the system.
    :return: None
    """
    logging.info("reboot...")
    system = platform.system()
    if system == 'Linux':
        os.system("sudo reboot -i")
    elif system == 'Windows':
        os.system("shutdown /r /t 1")
    else:
        raise Exception(f"Unsupported os '{system}'")
