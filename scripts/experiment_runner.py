# -*- coding: utf-8 -*-
"""
    Purpose: Run large scale experiments with EvoSuite
    Author: Kevin Haack (based on the batch script from Mitchell Olsthoorn)
"""
import argparse
import glob
import logging
import os
import random
import re
import signal
import socket
import subprocess
from datetime import datetime, timedelta

import psutil

import experiment_lib as ex

# files and folders
DIRECTORY_EXECUTION_REPORTS = "reports"
DIRECTORY_EXECUTION_TESTS = "tests"
DIRECTORY_EXECUTION_LOGS = "logs"
DIRECTORY_RESULTS = "results"
FILE_SELECTED_SAMPLE = "sample.txt"
FILE_NOT_SELECTED_SAMPLE = "notInSample.txt"
FILE_LOG = "output.log"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
# number of executions per class
EXECUTIONS_PER_CLASS = 10
# skip executions after x timeouts
SKIP_AFTER_TIMEOUTS = 2
# select randomly
RANDOM = True
# number of classes that should be selected
SAMPLE_SIZE = 119
# search budget in seconds
SEARCH_BUDGET = 120
# timeout in seconds
TIMEOUT = 180
# Algorithms (supported in this script: 'RANDOM' or 'DYNAMOSA')
ALGORITHM = 'DYNAMOSA'
# Criterion list as string (seperated with ':') or 'default' for EvoSuits default setting
CRITERION = 'default'
# Mutation rate as float or 'default'
MUTATION_RATE = 'default'
# Cross over rate as float or 'default'
CROSS_OVER_RATE = 'default'
# Parameters
PARAMETER_ALL = [
    '-Dshow_progress=false',
    '-Dplot=false',
    '-Dclient_on_thread=false',
    '-Dtrack_boolean_branches=true',
    '-Dtrack_covered_gradient_branches=true',
    '-Dsearch_budget=' + str(SEARCH_BUDGET)
]
PARAMETER_RANDOM = [
    '-generateRandom',
    '-Dalgorithm=RANDOM_SEARCH',
    '-Doutput_variables='
    'Algorithm,'
    'TARGET_CLASS,'
    'Generations,'
    'criterion,'
    'Coverage,'
    'Fitness,'
    'BranchCoverage,'
    'Total_Goals,'
    'Covered_Goals,'
    'Total_Time,'
    'Total_Branches,'
    'Covered_Branches,'
    'Lines,'
    'Covered_Lines'
]
PARAMETER_DYNAMOSA = [
    '-generateMOSuite',
    '-Dalgorithm=DYNAMOSA',
    '-Denable_fitness_history=true',
    '-Denable_landscape_analysis=true',
    '-Dtrack_boolean_branches=true',
    '-Dtrack_covered_gradient_branches=true',
    '-Doutput_variables='
    'Algorithm,'
    'TARGET_CLASS,'
    'Generations,'
    'criterion,'
    'Coverage,'
    'Fitness,'
    'BranchCoverage,'
    'Total_Goals,'
    'Covered_Goals,'
    'Total_Time,'
    'Total_Branches,'
    'Covered_Branches,'
    'Gradient_Branches,'
    'Gradient_Branches_Covered,'
    'Lines,'
    'Covered_Lines,'
    '_FitnessMax,'
    '_FitnessMin,'
    '_NeutralityVolume,'
    '_InformationContent,'
    '_FitnessRatio,'
    '_Generations,'
    '_GradientBranches,'
    '_GradientBranchesCovered'
]


def get_project_class_path(project):
    """
    Determines the classpath based on the project and outputs this.
    Expects the following file structure: projects/<project>/<jars>
    :param project: The project.
    Returns: Colon seperated class path
    """
    project_path = os.path.join(args.corpus, project)
    logging.debug("create projectClassPath for folder '" + project_path + "'")

    # add dependencies
    jar_list = glob.iglob(os.path.join(project_path, 'lib', '*.jar'), recursive=True)
    class_path = ""
    for jar in jar_list:
        class_path = class_path + jar + os.pathsep

    # add tested jar
    jar_list = glob.iglob(os.path.join(project_path, '*.jar'))
    for jar in jar_list:
        class_path = class_path + jar + os.pathsep

    return class_path


def move_results(project, clazz, path_class_dir, current_execution, path_results):
    """
    Move the exection results to the pathResults.
    :param project: The project.
    :param clazz: The current class.
    :param path_class_dir: The path of the class directory.
    :param current_execution: The index of the current execution.
    :param path_results: the destination of the results.
    :return: None
    """
    old_file_path = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(current_execution), "statistics.csv")

    newname = f"{project}-{clazz}-{str(current_execution)}.csv"
    new_file_path = os.path.join(path_results, newname)

    os.rename(old_file_path, new_file_path)


def create_parameter(project, clazz, path_class_dir, current_execution):
    """
    Create the EvoSuite parameter list for the passt parameter.
    :param project: The project.
    :param clazz: The class to test
    :param path_class_dir: The path to the class
    :param current_execution: The current execution index
    :return: The parameter for EvoSuite.
    """
    project_class_path = get_project_class_path(project)
    path_report = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(current_execution))
    path_test = os.path.join(path_class_dir, DIRECTORY_EXECUTION_TESTS, str(current_execution))

    parameter = ['java',
                 '-Xmx4G',
                 '-jar',
                 args.evosuite,
                 '-class',
                 clazz,
                 '-projectCP',
                 project_class_path,
                 f'-Dreport_dir={path_report}',
                 f'-Dtest_dir={path_test}'
                 ]

    parameter = parameter + PARAMETER_ALL

    if CRITERION != 'default':
        parameter = parameter + ['-criterion', CRITERION]

    if CROSS_OVER_RATE != 'default':
        parameter = parameter + ['-Dcrossover_rate', str(CROSS_OVER_RATE)]

    if MUTATION_RATE != 'default':
        parameter = parameter + ['-Dmutation_rate', str(MUTATION_RATE)]

    if ALGORITHM == 'DYNAMOSA':
        return parameter + PARAMETER_DYNAMOSA
    elif ALGORITHM == 'RANDOM':
        return parameter + PARAMETER_RANDOM
    else:
        raise ValueError("unsupported algorithm: " + ALGORITHM)


def run_evosuite(project, clazz, current_class, path_results):
    """
    Runs multiple executions of EvoSuite for the passed class.
    :param path_results: The path to the results directory
    :param project: The project of the class.
    :param clazz: The class to test.
    :param current_class: The index of the passed class.
    """
    timeouts = 0
    execution = 0
    skip = False
    while not skip and execution < EXECUTIONS_PER_CLASS:
        logging.info(
            f"Class ({str(current_class + 1)} / {str(SAMPLE_SIZE)}) Execution ({str(execution + 1)} / {str(EXECUTIONS_PER_CLASS)}): Running default configuration in project ({project}) for class ({clazz}) with random seed.")

        # output directories
        path_class_dir = os.path.join(args.corpus, project, clazz)

        # create directories
        if not os.path.exists(path_class_dir):
            os.mkdir(path_class_dir)

        # build evoSuite parameters
        parameter = create_parameter(project, clazz, path_class_dir, execution)

        # setup log
        path_log = os.path.join(path_class_dir, DIRECTORY_EXECUTION_LOGS)

        if not os.path.exists(path_log):
            os.mkdir(path_log)

        path_log_file = os.path.join(path_log, "log_" + str(execution) + ".txt")
        output = open(path_log_file, "w")

        # start process
        proc = subprocess.Popen(parameter, stdout=output, stderr=output)

        try:
            proc.communicate(timeout=TIMEOUT)
            move_results(project, clazz, path_class_dir, execution, path_results)
            timeouts = 0
        except subprocess.TimeoutExpired:
            # skip if timeouts reached
            timeouts = timeouts + 1
            if 0 < SKIP_AFTER_TIMEOUTS <= timeouts:
                skip = True
                logging.info(f"max timeouts reached, skip next")

            # kill process
            logging.warning(f'Subprocess timeout ({str(timeouts)}/{str(SKIP_AFTER_TIMEOUTS)}) {str(TIMEOUT)}s')
            kill_process(proc)
        except Exception as error:
            logging.error(f"Unexpected {error=}, {type(error)=}")

        execution = execution + 1


def kill_process(proc):
    """
    Kills the passed process and its subprocesses.
    :param proc: The process to kill.
    :return:
    """
    # killing child processes. The subprocess timeout does not work with child processes
    # https://stackoverflow.com/questions/36952245/subprocess-timeout-failure
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    process = psutil.Process(proc.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def on_timeout(process_arguments):
    """
    Function to be called on a thread timeout.
    :param process_arguments: at index 0, the proccess
    :return:
    """
    if process_arguments[0].pid >= 0:
        logging.error("Timeout after " + str(TIMEOUT) + "s: kill process " + args[0].pid)
        os.kill(process_arguments[0].pid, signal.SIGTERM)


def select_sample(init_sample):
    """
    Select a sample given the constants.
    :param init_sample: The initial sample of all classes.
    :return: The selected sample.
    """
    if RANDOM:
        return random.sample(range(len(init_sample)), SAMPLE_SIZE)
    else:
        return range(0, SAMPLE_SIZE)


def create_backups(initial_sample, sample):
    """
    Saves a backup of the selected samples.
    :param initial_sample: The initial sample of all classes.
    :param sample: The selected sample.
    :return:
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    selected_sample_path = os.path.join(script_path, FILE_SELECTED_SAMPLE)
    selected_not_sample_path = os.path.join(script_path, FILE_NOT_SELECTED_SAMPLE)

    file_sample = open(selected_sample_path, 'w')
    file_not_in_sample = open(selected_not_sample_path, 'w')
    for i in range(0, len(initial_sample)):
        line = initial_sample[i]
        if i in sample:
            file_sample.writelines(line[0] + '\t' + line[1] + '\n')
        else:
            file_not_in_sample.writelines(line[0] + '\t' + line[1] + '\n')

    file_sample.close()
    file_not_in_sample.close()


def get_initial_sample(sample_file_path):
    """
    Read the initial sample from the file.
    :param sample_file_path: The sample file path.
    :return: The initial sample.
    """
    init_sample = []
    file_init = open(sample_file_path, "r")

    for line in file_init:
        parts = re.split('\t', line)
        if len(parts) >= 2:
            init_sample.append((parts[0], parts[1].replace('\n', '')))

    file_init.close()
    return init_sample


def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    argument_parser = argparse.ArgumentParser(
        description="Run large scale experiments with EvoSuite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument("-sample", help="The path of the sample file", type=ex.file_path, required=True)
    argument_parser.add_argument("-corpus", help="The path of the corpus directory", type=ex.dir_path, required=True)
    argument_parser.add_argument("-evosuite", help="The path of the evosuite jar", type=ex.file_path, required=True)

    group = argument_parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-shutdown", help="Shutdown after the executions", action='store_true')
    group.add_argument("-reboot", help="Reboot after the executions", action='store_true')

    return argument_parser


def main():
    """
    Runs large scale experiment.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    log_file_path = os.path.join(script_path, FILE_LOG)
    logging.basicConfig(filename=log_file_path, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    logging.info("Initial sample file:\t" + args.sample)
    sample_list = get_initial_sample(args.sample)

    logging.info("Total number of classes:\t" + str(len(sample_list)))
    logging.info("Random sample selection:\t" + str(RANDOM))
    logging.info("Sample size:\t\t" + str(SAMPLE_SIZE))
    logging.info("Executions/Class:\t" + str(EXECUTIONS_PER_CLASS))
    logging.info("Search budget:\t\t" + str(SEARCH_BUDGET) + "s")
    logging.info("Timeout:\t\t\t" + str(TIMEOUT) + "s")
    logging.info("Skip after timeouts:\t" + str(SKIP_AFTER_TIMEOUTS))
    logging.info("Algorithm:\t\t" + ALGORITHM)
    logging.info("Criterion:\t\t" + CRITERION)
    logging.info("Mutation rate:\t\t" + str(MUTATION_RATE))
    logging.info("Cross over rate:\t\t" + str(CROSS_OVER_RATE))
    logging.info("Host:\t\t\t" + socket.gethostname())

    runtime = SAMPLE_SIZE * EXECUTIONS_PER_CLASS * SEARCH_BUDGET
    estimation = datetime.now() + timedelta(seconds=runtime)
    logging.info("Run time estimation:\t" + str(runtime / 60) + "min (end at " + str(estimation) + ")")

    if SAMPLE_SIZE > len(sample_list):
        logging.error("sample size '" + str(SAMPLE_SIZE) + "' > init file length '" + str(len(sample_list)) + "'")
        return

    # select sample
    sample = select_sample(sample_list)

    # save backup
    create_backups(sample_list, sample)

    # create result directory
    path_results = os.path.join(script_path, DIRECTORY_RESULTS)
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    now = datetime.now()
    path_results = os.path.join(script_path, DIRECTORY_RESULTS, now.strftime("%Y-%m-%d %H-%M-%S"))
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    # run tests
    for i in range(len(sample)):
        project = sample_list[sample[i]][0]
        clazz = sample_list[sample[i]][1]
        run_evosuite(project, clazz, i, path_results)

    logging.info("DONE.")
    if args.shutdown:
        ex.shutdown()
    elif args.reboot:
        ex.reboot()


if __name__ == "__main__":
    args = setup_argparse().parse_args()
    main()
