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
import time
from datetime import datetime

import numpy as np
import psutil

import experiment_lib as ex

# files and folders
DIRECTORY_EXECUTION_REPORTS = "reports"
DIRECTORY_EXECUTION_TESTS = "tests"
DIRECTORY_EXECUTION_LOGS = "logs"
DIRECTORY_RESULTS = "results"
FILE_SELECTED_SAMPLE = "sample.txt"
FILE_NOT_SELECTED_SAMPLE = "notInSample.txt"
FILE_STATUS = "status.log"
FILE_LOG = "output.log"
RESULT_DIR_FORMAT = "%Y-%m-%d %H-%M-%S"
# mease the mean runtime very x executions
MEASURE_MEAN_RUNTIME_EVERY = 10
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
# Parameters
PARAMETER_ALL = [
    '-Dshow_progress=false',
    '-Dplot=false',
    '-Dclient_on_thread=false',
    '-Dtrack_boolean_branches=true',
    '-Dtrack_covered_gradient_branches=true'
]
PARAMETER_RANDOM = [
    '-generateRandom',
    '-Dalgorithm=RANDOM_SEARCH',
    '-Doutput_variables='
    'Algorithm,'
    'TARGET_CLASS,'
    'Generations,'
    'Mutants,'
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
    '-Dtrack_boolean_branches=true',
    '-Dtrack_covered_gradient_branches=true',
    '-Doutput_variables='
    'Algorithm,'
    'TARGET_CLASS,'
    'Generations,'
    'Mutants,'
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
    '_GradientBranchesCovered,'
    '_ParameterControlled'
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


def remove_results(path_class_dir):
    """
    Remove the execution results.
    :param path_class_dir: The path of the class directory.
    :return: None
    """
    file_path = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(runner.current_execution),
                             "statistics.csv")

    if os.path.exists(file_path):
        os.remove(file_path)


def move_results(path_class_dir, path_results):
    """
    Move the execution results to the pathResults.
    :param path_class_dir: The path of the class directory.
    :param path_results: the destination of the results.
    :return: None
    """
    old_file_path = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(runner.current_execution),
                                 "statistics.csv")

    newname = f"{runner.current_project}-{runner.current_class}-{str(runner.current_execution)}.csv"
    new_file_path = os.path.join(path_results, newname)

    os.rename(old_file_path, new_file_path)


def create_parameter(path_class_dir):
    """
    Create the EvoSuite parameter list for the passt parameter.
    :param path_class_dir: The path to the class
    :return: The parameter for EvoSuite.
    """
    project_class_path = get_project_class_path(runner.current_project)
    path_report = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(runner.current_execution))
    path_test = os.path.join(path_class_dir, DIRECTORY_EXECUTION_TESTS, str(runner.current_execution))

    parameter = ['java',
                 '-Xmx4G',
                 '-jar',
                 args.evosuite,
                 '-class',
                 runner.current_class,
                 '-projectCP',
                 project_class_path,
                 f'-Dreport_dir={path_report}',
                 f'-Dtest_dir={path_test}',
                 f'-Dsearch_budget={str(runner.search_budget)}'
                 ]

    parameter = parameter + PARAMETER_ALL

    for key, value in runner.additional_parameter.items():
        if value is not None:
            parameter = parameter + [key + '=' + str(value)]

    if runner.algorithm == 'DYNAMOSA':
        return parameter + PARAMETER_DYNAMOSA
    elif runner.algorithm == 'RANDOM':
        return parameter + PARAMETER_RANDOM
    else:
        raise ValueError("unsupported algorithm: " + runner.algorithm)


def write_status_file(status):
    """
    Write the status file.
    :return: None
    """
    runner.status = status
    runner.saved_at = datetime.now()

    status_file_path = os.path.join(ex.get_script_path(), FILE_STATUS)
    with open(status_file_path, 'w') as status_file:
        runner.save_to_file(status_file)


def run_execution(path_results, parameter, output, path_class_dir):
    """
    Runs one execution of EvoSuite for the passed class.
    :param path_results: The path of the results.
    :param parameter: The parameter for the run.
    :param output: The output stream of the execution.
    :param path_class_dir: The class path.
    :return: True if succeeded
    """
    remove_results(path_class_dir)
    proc = subprocess.Popen(parameter, stdout=output, stderr=output)

    try:
        proc.communicate(timeout=runner.timeout)
        move_results(path_class_dir, path_results)

        return True
    except subprocess.TimeoutExpired:
        # kill process
        logging.warning(
            f'Subprocess timeout after {str(runner.timeout)}s')
        kill_process(proc)

        return False
    except Exception as error:
        logging.error(f"Unexpected {error=}, {type(error)=}")
        return False


def run_executions(path_results):
    """
    Runs multiple executions of EvoSuite for the passed class.
    :param path_results: The path to the results directory
    :return: None
    """
    attempts = 0
    runner.current_execution = 0
    skip = False
    measured_runtimes = []

    while not skip and runner.current_execution < runner.executions_per_class:
        start_time = time.time()

        logging.info(
            f"Class ({str(runner.current_class_index + 1)} / {str(runner.sample_size)}) Execution ({str(runner.current_execution + 1)} / {str(runner.executions_per_class)}): Running default configuration in project ({runner.current_project}) for class ({runner.current_class}) with random seed {runner.random}.")

        # write status
        if args.write_status:
            write_status_file(ex.Status.RUNNING)

        # output directories
        path_class_dir = os.path.join(args.corpus, runner.current_project, runner.current_class)

        # create directories
        if not os.path.exists(path_class_dir):
            os.mkdir(path_class_dir)

        # build evoSuite parameters
        parameter = create_parameter(path_class_dir)

        # setup log
        path_log = os.path.join(path_class_dir, DIRECTORY_EXECUTION_LOGS)

        if not os.path.exists(path_log):
            os.mkdir(path_log)

        path_log_file = os.path.join(path_log, "log_" + str(runner.current_execution) + ".txt")
        output = open(path_log_file, "w")

        # start process
        success = run_execution(path_results, parameter, output, path_class_dir)

        # skip?
        if success:
            attempts = 0
        else:
            attempts = attempts + 1
            if 0 < runner.number_attempts <= attempts:
                skip = True
                logging.info(f"max attempts reached, skip class")

        # measure runtime
        end_time = time.time()
        elapsed_time = end_time - start_time
        measured_runtimes.append(elapsed_time)

        if (len(measured_runtimes) % MEASURE_MEAN_RUNTIME_EVERY) == 0:
            runner.mean_runtime_per_execution = np.mean(measured_runtimes)

        # next
        runner.current_execution = runner.current_execution + 1


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
        logging.error("Timeout after " + str(runner.timeout) + "s: kill process " + args[0].pid)
        os.kill(process_arguments[0].pid, signal.SIGTERM)


def select_sample(init_sample):
    """
    Select a sample given the constants.
    :param init_sample: The initial sample of all classes.
    :return: The selected sample.
    """
    if runner.random:
        return random.sample(range(len(init_sample)), runner.sample_size)
    else:
        return range(0, runner.sample_size)


def create_backups(initial_sample, sample):
    """
    Saves a backup of the selected samples.
    :param initial_sample: The initial sample of all classes.
    :param sample: The selected sample.
    :return:
    """
    selected_sample_path = os.path.join(ex.get_script_path(), FILE_SELECTED_SAMPLE)
    selected_not_sample_path = os.path.join(ex.get_script_path(), FILE_NOT_SELECTED_SAMPLE)

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
    argument_parser.add_argument("-sample", help="The path of the sample file", type=ex.check_file_path, required=True)
    argument_parser.add_argument("-corpus", help="The path of the corpus directory", type=ex.check_dir_path,
                                 required=True)
    argument_parser.add_argument("-evosuite", help="The path of the evosuite jar", type=ex.check_file_path,
                                 required=True)
    argument_parser.add_argument("-executions", help="Number of executions per class", type=ex.check_positive_int,
                                 required=True)
    argument_parser.add_argument("-write_status", help="Write the status in the status file", action='store_true')

    group = argument_parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-shutdown", help="Shutdown after the executions", action='store_true')
    group.add_argument("-reboot", help="Reboot after the executions", action='store_true')

    return argument_parser


def main():
    """
    Runs large scale experiment.
    """
    log_file_path = os.path.join(ex.get_script_path(), FILE_LOG)
    logging.basicConfig(filename=log_file_path, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    sample_list = get_initial_sample(args.sample)

    if runner.sample_size is None:
        runner.sample_size = len(sample_list)

    runner.print_status()

    if runner.sample_size > len(sample_list):
        raise ValueError(f"sample size '{str(runner.sample_size)}' > init file length '{str(len(sample_list))}'")

    if args.write_status:
        write_status_file(ex.Status.IDLE)

    # select sample
    sample = select_sample(sample_list)

    # save backup
    create_backups(sample_list, sample)

    # create result directory
    path_results = os.path.join(ex.get_script_path(), DIRECTORY_RESULTS)
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    path_results = os.path.join(ex.get_script_path(), DIRECTORY_RESULTS,
                                runner.start_time.strftime(RESULT_DIR_FORMAT))
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    # run tests
    logging.info("run tests...")
    if args.write_status:
        write_status_file(ex.Status.RUNNING)

    try:
        for i in range(len(sample)):
            runner.current_class = sample_list[sample[i]][1]
            runner.current_project = sample_list[sample[i]][0]
            runner.current_class_index = i

            run_executions(path_results)
    except Exception as error:
        logging.error(f"Unexpected {error=}, {type(error)=}")
        if args.write_status:
            write_status_file(ex.Status.ERROR)

    if args.write_status:
        write_status_file(ex.Status.DONE)
    logging.info("DONE.")

    if args.shutdown:
        ex.shutdown()
    elif args.reboot:
        ex.reboot()


if __name__ == "__main__":
    args = setup_argparse().parse_args()

    # NONE
    # HIGH_STDEV
    # LOW_END_COVERAGE
    # RELATIVE_LOW_COVERAGE
    # HIGH_STDEV_LOW_END_COVERAGE
    # HIGH_STDEV_RELATIVE_LOW_COVERAGE
    # ALWAYS
    # HIGHER_WITH_POP125_and_RELATIVE_LOW_COVERAGE

    additional_parameter = {
        '-criterion': None,
        '-Dcrossover_rate': None,
        '-Dmutation_rate': None,
        '-Dpopulation': 25,
        '-Dp_test_insertion': 1.0,
        '-Dnumber_of_mutations': None,
        '-Denable_landscape_analysis': 'true',
        '-Denable_fitness_history': 'true',
        '-Denable_parameter_control': 'false',
        '-Drandom_seed': 42,
        '-Dpc_at': 0.3,
        '-Dpc_population': None,
        '-Dpc_p_test_insertion': None,
        '-Dpc_crossover_rate': None,
        '-Dpc_number_of_mutations': None,
        '-Dpc_rank_bias': None,
        '-Dpc_prediction': 'NONE'
    }

    runner = ex.ExperimentRunner(initial_sample_file=args.sample,
                                 executions_per_class=args.executions,
                                 hostname=socket.gethostname(),
                                 start_time=datetime.now(),
                                 random=False,
                                 additional_parameter=additional_parameter)
    main()
