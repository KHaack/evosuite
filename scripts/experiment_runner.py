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
from datetime import datetime

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


def move_results(path_class_dir, path_results):
    """
    Move the exection results to the pathResults.
    :param path_class_dir: The path of the class directory.
    :param path_results: the destination of the results.
    :return: None
    """
    old_file_path = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(runner_status.current_execution),
                                 "statistics.csv")

    newname = f"{runner_status.current_project}-{runner_status.current_class}-{str(runner_status.current_execution)}.csv"
    new_file_path = os.path.join(path_results, newname)

    os.rename(old_file_path, new_file_path)


def create_parameter(path_class_dir):
    """
    Create the EvoSuite parameter list for the passt parameter.
    :param path_class_dir: The path to the class
    :return: The parameter for EvoSuite.
    """
    project_class_path = get_project_class_path(runner_status.current_project)
    path_report = os.path.join(path_class_dir, DIRECTORY_EXECUTION_REPORTS, str(runner_status.current_execution))
    path_test = os.path.join(path_class_dir, DIRECTORY_EXECUTION_TESTS, str(runner_status.current_execution))

    parameter = ['java',
                 '-Xmx4G',
                 '-jar',
                 args.evosuite,
                 '-class',
                 runner_status.current_class,
                 '-projectCP',
                 project_class_path,
                 f'-Dreport_dir={path_report}',
                 f'-Dtest_dir={path_test}',
                 f'-Dsearch_budget={str(runner_status.search_budget)}'
                 ]

    parameter = parameter + PARAMETER_ALL

    if runner_status.criterion != 'default':
        parameter = parameter + ['-criterion', runner_status.criterion]

    if runner_status.cross_over_rate != 'default':
        parameter = parameter + ['-Dcrossover_rate', str(runner_status.cross_over_rate)]

    if runner_status.mutation_rate != 'default':
        parameter = parameter + ['-Dmutation_rate', str(runner_status.mutation_rate)]

    if runner_status.algorithm == 'DYNAMOSA':
        return parameter + PARAMETER_DYNAMOSA
    elif runner_status.algorithm == 'RANDOM':
        return parameter + PARAMETER_RANDOM
    else:
        raise ValueError("unsupported algorithm: " + runner_status.algorithm)


def write_status_file():
    """
    Write the status file.
    :return: None
    """
    status_file_path = os.path.join(ex.get_script_path(), FILE_STATUS)
    with open(status_file_path, 'w') as status_file:
        runner_status.save_to_file(status_file)


def run_evosuite(path_results):
    """
    Runs multiple executions of EvoSuite for the passed class.
    :param path_results: The path to the results directory
    """
    timeouts = 0
    runner_status.current_execution = 0
    skip = False
    while not skip and runner_status.current_execution < runner_status.executions_per_class:
        logging.info(
            f"Class ({str(runner_status.current_class_index + 1)} / {str(runner_status.sample_size)}) Execution ({str(runner_status.current_execution + 1)} / {str(runner_status.executions_per_class)}): Running default configuration in project ({runner_status.current_project}) for class ({runner_status.current_class}) with random seed.")

        # write status
        if args.write_status:
            write_status_file()

        # output directories
        path_class_dir = os.path.join(args.corpus, runner_status.current_project, runner_status.current_class)

        # create directories
        if not os.path.exists(path_class_dir):
            os.mkdir(path_class_dir)

        # build evoSuite parameters
        parameter = create_parameter(path_class_dir)

        # setup log
        path_log = os.path.join(path_class_dir, DIRECTORY_EXECUTION_LOGS)

        if not os.path.exists(path_log):
            os.mkdir(path_log)

        path_log_file = os.path.join(path_log, "log_" + str(runner_status.current_execution) + ".txt")
        output = open(path_log_file, "w")

        # start process
        proc = subprocess.Popen(parameter, stdout=output, stderr=output)

        try:
            proc.communicate(timeout=runner_status.timeout)
            move_results(path_class_dir, path_results)
            timeouts = 0
        except subprocess.TimeoutExpired:
            # skip if timeouts reached
            timeouts = timeouts + 1
            if 0 < runner_status.skip_after_timeouts <= timeouts:
                skip = True
                logging.info(f"max timeouts reached, skip next")

            # kill process
            logging.warning(
                f'Subprocess timeout ({str(timeouts)}/{str(runner_status.skip_after_timeouts)}) {str(runner_status.timeout)}s')
            kill_process(proc)
        except Exception as error:
            logging.error(f"Unexpected {error=}, {type(error)=}")

        runner_status.current_execution = runner_status.current_execution + 1


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
        logging.error("Timeout after " + str(runner_status.timeout) + "s: kill process " + args[0].pid)
        os.kill(process_arguments[0].pid, signal.SIGTERM)


def select_sample(init_sample):
    """
    Select a sample given the constants.
    :param init_sample: The initial sample of all classes.
    :return: The selected sample.
    """
    if runner_status.random:
        return random.sample(range(len(init_sample)), runner_status.sample_size)
    else:
        return range(0, runner_status.sample_size)


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
    argument_parser.add_argument("-sample", help="The path of the sample file", type=ex.file_path, required=True)
    argument_parser.add_argument("-corpus", help="The path of the corpus directory", type=ex.dir_path, required=True)
    argument_parser.add_argument("-evosuite", help="The path of the evosuite jar", type=ex.file_path, required=True)
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

    runner_status.print_status()

    if runner_status.sample_size > len(sample_list):
        raise ValueError(f"sample size '{str(runner_status.sample_size)}' > init file length '{str(len(sample_list))}'")

    # select sample
    sample = select_sample(sample_list)

    # save backup
    create_backups(sample_list, sample)

    # create result directory
    path_results = os.path.join(ex.get_script_path(), DIRECTORY_RESULTS)
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    path_results = os.path.join(ex.get_script_path(), DIRECTORY_RESULTS,
                                runner_status.start_time.strftime(RESULT_DIR_FORMAT))
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    # write status
    if args.write_status:
        write_status_file()

    # run tests
    logging.info("run tests...")
    for i in range(len(sample)):
        runner_status.current_class = sample_list[sample[i]][1]
        runner_status.current_project = sample_list[sample[i]][0]
        runner_status.current_class_index = i
        run_evosuite(path_results)

    logging.info("DONE.")
    if args.shutdown:
        ex.shutdown()
    elif args.reboot:
        ex.reboot()


if __name__ == "__main__":
    args = setup_argparse().parse_args()
    now = datetime.now()
    runner_status = ex.RunnerStatus(initial_sample_file=args.sample, sample_size=713, executions_per_class=5,
                                    hostname=socket.gethostname(), start_time=now, random=False)
    main()
