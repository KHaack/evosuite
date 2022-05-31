"""
    Purpose: This experiment library contains shared functions for the experiments.
    Author: Kevin Haack
"""
import argparse
import json
import logging
import os
import platform
from datetime import datetime
from enum import IntEnum

import pandas as pd
from dateutil import parser

# files and directories
FILE_CLASSES = "samples\\00 - original - 23894.txt"
FILE_EXPORT = "export.txt"
# filtering
FILTER_ZERO_GENERATIONS = True
FILTER_PERCENTAGE = True


def get_script_path():
    """
    Returns the script path.
    :return: Returns the script path.
    """
    return os.path.dirname(os.path.realpath(__file__))


def create_export(dataframe):
    """
    Export the passed dataframe to the runner format.
    :param dataframe: The dataframe to export
    :return: None
    """
    path_samples = os.path.join(get_script_path(), FILE_CLASSES)
    path_export = os.path.join(get_script_path(), FILE_EXPORT)

    samples = pd.read_csv(path_samples, delimiter='\t', names=['project', 'TARGET_CLASS'])
    merged = dataframe.merge(samples, left_on='TARGET_CLASS', right_on='TARGET_CLASS', how='inner')
    merged = merged.groupby(['project', 'TARGET_CLASS']).count().reset_index()[['project', 'TARGET_CLASS']]

    merged.to_csv(path_export, header=None, sep='\t', index=False)


def add_additional_columns(dataframe):
    dataframe['PercentageReached'] = dataframe['NeutralityVolume'].str.count(';') * 10

    # classes
    # classification groundTruth
    dataframe['GroundTruth'] = dataframe['Coverage'].ge(0.8)

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

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_NotGradRatio'] = 1
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_NotGradRatio'] = (dataframe['Total_Branches'] - dataframe[
        '_GradientBra']) / (dataframe['Total_Branches'])

    dataframe.loc[dataframe['_GradientBra'].eq(0), '_GradientRatio'] = 1
    dataframe.loc[dataframe['_GradientBra'].gt(0), '_GradientRatio'] = dataframe['_GradientCov'] / (
        dataframe['_GradientBra'])

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_BranchRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_BranchRatio'] = dataframe['_GradientBra'] / (
        dataframe['Total_Branches'])

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


class Status(IntEnum):
    UNKNOWN = 0
    IDLE = 1
    RUNNING = 2
    DONE = 3
    ERROR = 4


class ExperimentRunner:
    """
    Represents the experiment runner.
    """

    def __init__(self, initial_sample_file, hostname, start_time, sample_size, random=True, executions_per_class=10,
                 search_budget=120, timeout=180,
                 skip_after_timeouts=2, algorithm='DYNAMOSA', criterion='default', mutation_rate='default',
                 cross_over_rate='default', current_project=None, current_class=None, current_class_index=0,
                 current_execution=0, status=Status.UNKNOWN):
        self.initial_sample_file = initial_sample_file
        self.random = random
        self.sample_size = sample_size
        self.executions_per_class = executions_per_class
        self.search_budget = search_budget
        self.timeout = timeout
        self.skip_after_timeouts = skip_after_timeouts
        self.algorithm = algorithm
        self.criterion = criterion
        self.mutation_rate = mutation_rate
        self.cross_over_rate = cross_over_rate
        self.hostname = hostname
        self.current_project = current_project
        self.current_class = current_class
        self.current_class_index = current_class_index
        self.current_execution = current_execution
        self.status = status

        if isinstance(start_time, str):
            self.start_time = parser.parse(start_time)
        else:
            self.start_time = start_time

    def print_status(self):
        """
        Prints the runner status
        :return: None
        """
        logging.info(f"Initial sample file:\t{self.initial_sample_file}")
        logging.info(f"Random sample selection:\t{str(self.random)}")
        logging.info(f"Sample size:\t\t{str(self.sample_size)}")
        logging.info(f"Executions/Class:\t{str(self.executions_per_class)}")
        logging.info(f"Search budget:\t\t{str(self.search_budget)}s")
        logging.info(f"Timeout:\t\t\t{str(self.timeout)}s")
        logging.info(f"Skip after timeouts:\t{str(self.skip_after_timeouts)}")
        logging.info(f"Algorithm:\t\t{self.algorithm}")
        logging.info(f"Criterion:\t\t{self.criterion}")
        logging.info(f"Mutation rate:\t\t{str(self.mutation_rate)}")
        logging.info(f"Cross over rate:\t\t{str(self.cross_over_rate)}")
        logging.info(f"Host:\t\t\t{self.hostname}")
        logging.info(f"Start time:\t\t{self.start_time.strftime('%Y-%m-%d %H-%M-%S')}")
        logging.info(f"Runtime estimation:\t{str(self.get_runtime_estimation() / 60 / 60)}h")

    def get_runtime_estimation(self):
        """
        Returns the runtime estimation.
        :return: Returns the runtime estimation.
        """
        estimation = (self.sample_size - self.current_class_index) * (
                    self.executions_per_class - 1)
        estimation = estimation + (self.executions_per_class - self.current_execution)
        return estimation * self.search_budget

    def save_to_file(self, status_file):
        """
        Write the object to the passed file object.
        :param status_file: The file object.
        :return: None
        """
        status_file.write(json.dumps(self.__dict__, indent=4, cls=ExperimentRunnerEncoder))


class ExperimentRunnerEncoder(json.JSONEncoder):
    """
    A Datetime aware encoder for json.
    https://stackoverflow.com/questions/44630103/how-to-write-and-read-datetime-dictionaries
    """

    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        else:
            return json.JSONEncoder.default(self, o)
