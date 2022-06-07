"""
    Purpose: This experiment library contains shared functions for the experiments.
    Author: Kevin Haack
"""
import argparse
import json
import logging
import os
import platform
from datetime import timedelta, datetime
from enum import IntEnum

import pandas as pd
from dateutil import parser

# files and directories
FILE_CLASSES = "C:\\Users\\kha\\Desktop\\Benchmark\\samples\\00 - original - 23894.txt"
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
    """
    Represents a runner status.
    """
    UNKNOWN = 0
    IDLE = 1
    RUNNING = 2
    DONE = 3
    ERROR = 4


class ExperimentRunner:
    """
    Represents the experiment runner, and it's status during the run.
    """

    def __init__(self, initial_sample_file, hostname, start_time, sample_size, random=True, executions_per_class=10,
                 search_budget=120, timeout=180,
                 number_attempts=2, algorithm='DYNAMOSA', criterion=None, mutation_rate=None,
                 cross_over_rate=None, current_project=None, current_class=None, current_class_index=0,
                 current_execution=0, status=Status.UNKNOWN, saved_at=None, mean_runtime_per_execution=None,
                 population=None):
        self.initial_sample_file = initial_sample_file
        self.random = random
        self.sample_size = sample_size
        self.executions_per_class = executions_per_class
        self.search_budget = search_budget
        self.timeout = timeout
        self.number_attempts = number_attempts
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
        self.population = population

        if mean_runtime_per_execution is None:
            self.mean_runtime_per_execution = self.search_budget
        else:
            self.mean_runtime_per_execution = mean_runtime_per_execution

        if isinstance(start_time, str):
            self.start_time = parser.parse(start_time)
        else:
            self.start_time = start_time

        if isinstance(saved_at, str):
            self.saved_at = parser.parse(saved_at)
        else:
            self.saved_at = saved_at

    def print_status(self):
        """
        Prints the runner status
        :return: None
        """
        if self.status == Status.UNKNOWN:
            logging.warning(f"{self.hostname} status:\tunknown")
        elif self.status == Status.IDLE:
            logging.info(f"{self.hostname} status:\tidle")
        elif self.status == Status.RUNNING:
            if self.saved_at is None:
                logging.warning(f"{self.hostname} status:\trunning (no saved_at)")
            else:
                # check saved_at
                delta = timedelta(seconds=self.timeout)
                last_accepted_time = self.saved_at + delta

                if datetime.now() >= last_accepted_time:
                    logging.info(f"{self.hostname} status:\trunning, BUT no new status file")

                else:
                    logging.info(f"{self.hostname} status:\trunning")
        elif self.status == Status.DONE:
            # done
            logging.info(f"{self.hostname} status:\tdone")
        elif self.status == Status.ERROR:
            # error
            logging.error(f"{self.hostname} status:\terror")

        logging.info(f"Initial sample file:\t{self.initial_sample_file}")
        logging.info(f"Random sample selection:\t{str(self.random)}")
        logging.info(f"Sample size:\t\t{str(self.sample_size)}")
        logging.info(f"Executions/Class:\t{str(self.executions_per_class)}")
        logging.info(f"Search budget:\t\t{str(self.search_budget)}s")
        logging.info(f"Mean runtime/execution:\t{str(self.mean_runtime_per_execution)}s")
        logging.info(f"Timeout:\t\t\t{str(self.timeout)}s")
        logging.info(f"Number of attempts:\t{str(self.number_attempts)}")
        logging.info(f"Algorithm:\t\t{self.algorithm}")
        logging.info(f"Host:\t\t\t{self.hostname}")

        if self.criterion is None:
            logging.info(f"Criterion:\t\tdefault")
        else:
            logging.info(f"Criterion:\t\t{self.criterion}")

        if self.mutation_rate is None:
            logging.info(f"Mutation rate:\t\tdefault")
        else:
            logging.info(f"Mutation rate:\t\t{str(self.mutation_rate)}")

        if self.cross_over_rate is None:
            logging.info(f"Crossover rate:\t\tdefault")
        else:
            logging.info(f"Crossover rate:\t\t{str(self.cross_over_rate)}")

        if self.population is None:
            logging.info(f"Population:\t\tdefault")
        else:
            logging.info(f"Population:\t\t{str(self.population)}")

        if self.start_time is None:
            logging.info(f"Start time:\t\t-")
        else:
            logging.info(f"Start time:\t\t{self.start_time.strftime('%Y-%m-%d %H-%M-%S')}")

        if self.saved_at is None:
            logging.info("Saved at:\t\t-")
        else:
            logging.info(f"Saved at:\t\t{self.saved_at.strftime('%Y-%m-%d %H-%M-%S')}")

        logging.info(f"Runtime estimation:\t{str(self.get_runtime_estimation() / 60 / 60)}h")

    def get_runtime_estimation(self):
        """
        Returns the runtime estimation.
        :return: Returns the runtime estimation.
        """
        if self.current_class_index is None or self.executions_per_class is None or self.current_execution is None:
            return 0

        executions = (self.current_class_index * self.executions_per_class) + self.current_execution
        executions_todo = (self.sample_size * self.executions_per_class) - executions

        return executions_todo * self.mean_runtime_per_execution

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
