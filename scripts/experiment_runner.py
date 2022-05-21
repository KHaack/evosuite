# -*- coding: utf-8 -*-
"""
    Purpose: Run large scale experiments with EvoSuite
    Author: Kevin Haack (based on the batch script from Mitchell Olsthoorn)
"""
import glob
import logging
import os
import random
import re
import signal
import socket
import subprocess
import sys
import psutil
from datetime import datetime, timedelta

# files and folders
DIRECTORY_CORPUS = "SF110-20130704"
DIRECTORY_REPORTS = "reports"
DIRECTORY_TESTS = "tests"
DIRECTORY_LOGS = "logs"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
PATH_EVOSUITE = "C:\\Users\\kha\\repos\\evosuite\\master\\target\\evosuite-master-1.2.1-SNAPSHOT.jar"
FILE_CLASSES_INIT = "init.txt"
FILE_CLASSES_SAMPLE = "sample.txt"
FILE_CLASSES_NOT_IN_SAMPLE = "notInSample.txt"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# number of executions per class
EXECUTIONS_PER_CLASS = 3
# skip executions after x timeouts
SKIP_AFTER_TIMEOUTS = 2
# select randomly
RANDOM = False
# number of classes that should be selected
SAMPLE_SIZE = 15
# search budget in seconds
SEARCH_BUDGET = 120
# timeout in seconds
TIMEOUT = 180
# Algorithms (supported in this script: 'RANDOM' or 'DYNAMOSA')
ALGORITHM = 'DYNAMOSA'
# Criterion list as string (seperated with ':') or 'default' for EvoSuits default setting
CRITERION = 'branch:line'
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
    projectPath = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, project)
    logging.debug("create projectClassPath for folder '" + projectPath + "'")

    jarList = glob.iglob(os.path.join(projectPath, '**/*.jar'), recursive=True)
    classPath = ""
    for jar in jarList:
        classPath = classPath + jar + os.pathsep

    return classPath


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
    pathReport = os.path.join(pathClassDir, DIRECTORY_REPORTS, str(currentExecution))
    pathTest = os.path.join(pathClassDir, DIRECTORY_TESTS, str(currentExecution))

    parameter = ['java',
                 '-Xmx4G',
                 '-jar',
                 PATH_EVOSUITE,
                 '-class',
                 clazz,
                 '-projectCP',
                 projectClassPath,
                 '-Dreport_dir=' + pathReport,
                 '-Dtest_dir=' + pathTest
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


def runEvoSuite(project, clazz, currentClass):
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
        pathClassDir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, project, clazz)

        # create directories
        if not os.path.exists(pathClassDir):
            os.mkdir(pathClassDir)

        # build evoSuite parameters
        parameter = createParameter(project, clazz, pathClassDir, execution)

        # setup log
        pathLog = os.path.join(pathClassDir, DIRECTORY_LOGS)

        if not os.path.exists(pathLog):
            os.mkdir(pathLog)

        pathLogFile = os.path.join(pathLog, "log_" + str(execution) + ".txt")
        output = open(pathLogFile, "w")

        # start process
        proc = subprocess.Popen(parameter, stdout=output, stderr=output)

        try:
            proc.communicate(timeout=TIMEOUT)
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
    :param files: The initial sample of all classes.
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
    pathSample = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, FILE_CLASSES_SAMPLE)
    pathNotInSample = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, FILE_CLASSES_NOT_IN_SAMPLE)

    fileSample = open(pathSample, 'w')
    fileNotInSample = open(pathNotInSample, 'w')
    for i in range(0, len(initSample)):
        line = initSample[i]
        if i in sample:
            fileSample.writelines(line[0] + '\t' + line[1] + '\n')
        else:
            fileNotInSample.writelines(line[0] + '\t' + line[1] + '\n')

    fileSample.close()
    fileNotInSample.close()


def getInitSample():
    """
    Read the initial sample from the file.
    :return: The initial sample.
    """
    pathInit = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, FILE_CLASSES_INIT)

    initSample = []
    fileInit = open(pathInit, "r")

    for line in fileInit:
        parts = re.split('\t', line)
        if (len(parts) >= 2):
            initSample.append((parts[0], parts[1].replace('\n', '')))

    fileInit.close()
    return initSample


def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    initSample = getInitSample()

    logging.info("Initial sample file:\t" + FILE_CLASSES_INIT)
    logging.info("Total number of classes:\t" + str(len(initSample)))
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

    if SAMPLE_SIZE > len(initSample):
        logging.error("sample size '" + str(SAMPLE_SIZE) + "' > init file length '" + str(len(initSample)) + "'")
        return

    # select sample
    sample = selectSample(initSample)

    # save backup
    createBackups(initSample, sample)

    # run tests
    for i in range(len(sample)):
        project = initSample[sample[i]][0]
        clazz = initSample[sample[i]][1]
        runEvoSuite(project, clazz, i)


if __name__ == "__main__":
    main()
