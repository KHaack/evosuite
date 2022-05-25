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
import experiment_lib as ex
from datetime import datetime, timedelta

import psutil

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
EXECUTIONS_PER_CLASS = 3
# skip executions after x timeouts
SKIP_AFTER_TIMEOUTS = 2
# select randomly
RANDOM = False
# number of classes that should be selected
SAMPLE_SIZE = 356
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


def getProjectClassPath(project):
    """
    Determines the classpath based on the project and outputs this.
    Expects the following file structure: projects/<project>/<jars>
    :param project: The project.
    Returns: Colon seperated class path
    """
    projectPath = os.path.join(args.corpus, project)
    logging.debug("create projectClassPath for folder '" + projectPath + "'")

    # add dependencies
    jarList = glob.iglob(os.path.join(projectPath, 'lib', '*.jar'), recursive=True)
    classPath = ""
    for jar in jarList:
        classPath = classPath + jar + os.pathsep

    # add tested jar
    jarList = glob.iglob(os.path.join(projectPath, '*.jar'))
    for jar in jarList:
        classPath = classPath + jar + os.pathsep

    return classPath


def moveResults(project, clazz, pathClassDir, currentExecution, pathResults):
    """
    Move the exection results to the pathResults.
    :param project: The project.
    :param clazz: The current class.
    :param pathClassDir: The path of the class directory.
    :param currentExecution: The index of the current execution.
    :param pathResults: the destination of the results.
    :return: None
    """
    oldFilePath = os.path.join(pathClassDir, DIRECTORY_EXECUTION_REPORTS, str(currentExecution), "statistics.csv")

    newname = f"{project}-{clazz}-{str(currentExecution)}.csv"
    newFilePath = os.path.join(pathResults, newname)

    os.rename(oldFilePath, newFilePath)


def createParameter(project, clazz, pathClassDir, currentExecution):
    """
    Create the EvoSuite parameter list for the passt parameter.
    :param project: The project.
    :param clazz: The class to test
    :param pathClassDir: The path to the class
    :param currentExecution: The current execution index
    :return: The parameter for EvoSuite.
    """
    projectClassPath = getProjectClassPath(project)
    pathReport = os.path.join(pathClassDir, DIRECTORY_EXECUTION_REPORTS, str(currentExecution))
    pathTest = os.path.join(pathClassDir, DIRECTORY_EXECUTION_TESTS, str(currentExecution))

    parameter = ['java',
                 '-Xmx4G',
                 '-jar',
                 args.evosuite,
                 '-class',
                 clazz,
                 '-projectCP',
                 projectClassPath,
                 f'-Dreport_dir={pathReport}',
                 f'-Dtest_dir={pathTest}'
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


def runEvoSuite(project, clazz, currentClass, pathResults):
    """
    Runs multiple executions of EvoSuite for the passed class.
    :param project: The project of the class.
    :param clazz: The class to test.
    :param currentClass: The index of the passed class.
    """
    timeouts = 0
    execution = 0
    skip = False
    while not skip and execution < EXECUTIONS_PER_CLASS:
        logging.info(
            f"Class ({str(currentClass + 1)} / {str(SAMPLE_SIZE)}) Execution ({str(execution + 1)} / {str(EXECUTIONS_PER_CLASS)}): Running default configuration in project ({project}) for class ({clazz}) with random seed.")

        # output directories
        pathClassDir = os.path.join(args.corpus, project, clazz)

        # create directories
        if not os.path.exists(pathClassDir):
            os.mkdir(pathClassDir)

        # build evoSuite parameters
        parameter = createParameter(project, clazz, pathClassDir, execution)

        # setup log
        pathLog = os.path.join(pathClassDir, DIRECTORY_EXECUTION_LOGS)

        if not os.path.exists(pathLog):
            os.mkdir(pathLog)

        pathLogFile = os.path.join(pathLog, "log_" + str(execution) + ".txt")
        output = open(pathLogFile, "w")

        # start process
        proc = subprocess.Popen(parameter, stdout=output, stderr=output)

        try:
            proc.communicate(timeout=TIMEOUT)
            moveResults(project, clazz, pathClassDir, execution, pathResults)
            timeouts = 0
        except subprocess.TimeoutExpired:
            # skip if timeouts reached
            timeouts = timeouts + 1
            if 0 < SKIP_AFTER_TIMEOUTS <= timeouts:
                skip = True
                logging.info(f"max timeouts reached, skip next")

            # kill process
            logging.warning(f'Subprocess timeout ({str(timeouts)}/{str(SKIP_AFTER_TIMEOUTS)}) {str(TIMEOUT)}s')
            killProcess(proc)
        except Exception as error:
            logging.error(f"Unexpected {error=}, {type(error)=}")

        execution = execution + 1


def killProcess(proc):
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


def onTimeout(args):
    """
    Function to be called on a thread timeout.
    :param args: at index 0, the proccess
    :return:
    """
    if args[0].pid >= 0:
        logging.error("Timeout after " + str(TIMEOUT) + "s: kill process " + args[0].pid)
        os.kill(args[0].pid, signal.SIGTERM)


def selectSample(initSample):
    """
    Select a sample given the constants.
    :param initSample: The initial sample of all classes.
    :return: The selected sample.
    """
    if RANDOM:
        return random.sample(range(len(initSample)), SAMPLE_SIZE)
    else:
        return range(0, SAMPLE_SIZE)


def createBackups(initSample, sample):
    """
    Saves a backup of the selected samples.
    :param initSample: The initial sample of all classes.
    :param sample: The selected sample.
    :return:
    """
    scriptPath = os.path.dirname(os.path.realpath(__file__))
    selectedSamplePath = os.path.join(scriptPath, FILE_SELECTED_SAMPLE)
    selectedNotSamplePath = os.path.join(scriptPath, FILE_NOT_SELECTED_SAMPLE)

    fileSample = open(selectedSamplePath, 'w')
    fileNotInSample = open(selectedNotSamplePath, 'w')
    for i in range(0, len(initSample)):
        line = initSample[i]
        if i in sample:
            fileSample.writelines(line[0] + '\t' + line[1] + '\n')
        else:
            fileNotInSample.writelines(line[0] + '\t' + line[1] + '\n')

    fileSample.close()
    fileNotInSample.close()


def getInitSample(sampleFilePath):
    """
    Read the initial sample from the file.
    :param sampleFilePath: The sample file path.
    :return: The initial sample.
    """
    initSample = []
    fileInit = open(sampleFilePath, "r")

    for line in fileInit:
        parts = re.split('\t', line)
        if len(parts) >= 2:
            initSample.append((parts[0], parts[1].replace('\n', '')))

    fileInit.close()
    return initSample


def setupArgparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(
        description="Run large scale experiments with EvoSuite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-sample", help="The path of the sample file", type=ex.file_path, required=True)
    parser.add_argument("-corpus", help="The path of the corpus directory", type=ex.dir_path, required=True)
    parser.add_argument("-evosuite", help="The path of the evosuite jar", type=ex.file_path, required=True)

    return parser


def main():
    """
    Runs large scale experiment.
    """
    scriptPath = os.path.dirname(os.path.realpath(__file__))
    logFilePath = os.path.join(scriptPath, FILE_LOG)
    logging.basicConfig(filename=logFilePath, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    logging.info("Initial sample file:\t" + args.sample)
    sampleList = getInitSample(args.sample)

    logging.info("Total number of classes:\t" + str(len(sampleList)))
    logging.info("Random sample selection:\t" + str(RANDOM))
    logging.info("Sample size:\t\t" + str(SAMPLE_SIZE))
    logging.info("Executions/Class:\t" + str(EXECUTIONS_PER_CLASS))
    logging.info("Search budget:\t\t" + str(SEARCH_BUDGET) + "s")
    logging.info("Timeout:\t\t\t" + str(TIMEOUT) + "s")
    logging.info("Algorithm:\t\t" + ALGORITHM)
    logging.info("Criterion:\t\t" + CRITERION)
    logging.info("Mutation rate:\t\t" + str(MUTATION_RATE))
    logging.info("Cross over rate:\t\t" + str(CROSS_OVER_RATE))
    logging.info("Host:\t\t\t" + socket.gethostname())

    runtime = SAMPLE_SIZE * EXECUTIONS_PER_CLASS * SEARCH_BUDGET
    estimation = datetime.now() + timedelta(seconds=runtime)
    logging.info("Run time estimation:\t" + str(runtime / 60) + "min (end at " + str(estimation) + ")")

    if SAMPLE_SIZE > len(sampleList):
        logging.error("sample size '" + str(SAMPLE_SIZE) + "' > init file length '" + str(len(sampleList)) + "'")
        return

    # select sample
    sample = selectSample(sampleList)

    # save backup
    createBackups(sampleList, sample)

    # create result directory
    pathResults = os.path.join(scriptPath, DIRECTORY_RESULTS)
    if not os.path.exists(pathResults):
        os.mkdir(pathResults)

    now = datetime.now()
    pathResults = os.path.join(scriptPath, DIRECTORY_RESULTS, now.strftime("%Y-%m-%d %H-%M-%S"))
    if not os.path.exists(pathResults):
        os.mkdir(pathResults)

    # run tests
    for i in range(len(sample)):
        project = sampleList[sample[i]][0]
        clazz = sampleList[sample[i]][1]
        runEvoSuite(project, clazz, i, pathResults)


if __name__ == "__main__":
    parser = setupArgparse()
    args = parser.parse_args()
    main()
