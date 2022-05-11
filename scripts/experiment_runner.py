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
import subprocess
import sys
import psutil

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
# tests
EXECUTIONS_PER_CLASS = 3
RANDOM = False
SAMPLE_SIZE = 91
# timing
SEARCH_BUDGET = 120
TIMEOUT = 240
# Algorithms and parameters (RANDOM or DYNAMOSA)
ALGORITHM = 'DYNAMOSA'
PARAMETER_ALL = [
    '-Dshow_progress=false',
    '-Dplot=false',
    '-Dclient_on_thread=false',
    '-criterion',
    'branch:line',
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
        classPath = classPath + jar + ";"

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
    for currentExecution in range(0, EXECUTIONS_PER_CLASS):
        logging.info("Class (" + str(currentClass + 1) + " / " + str(SAMPLE_SIZE) + ") Execution (" + str(
            currentExecution + 1) + " / " + str(
            EXECUTIONS_PER_CLASS) + "): Running default configuration for class (" + clazz + ") in project (" + project + ") with random seed.")

        # output directories
        pathClassDir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, project, clazz)

        # create directories
        if not os.path.exists(pathClassDir):
            os.mkdir(pathClassDir)

        # build evoSuite parameters
        parameter = createParameter(project, clazz, pathClassDir, currentExecution)

        # setup log
        pathLog = os.path.join(pathClassDir, DIRECTORY_LOGS)

        if not os.path.exists(pathLog):
            os.mkdir(pathLog)

        pathLogFile = os.path.join(pathLog, "log_" + str(currentExecution) + ".txt")
        output = open(pathLogFile, "w")

        # start process
        proc = subprocess.Popen(parameter, stdout=output, stderr=output)

        try:
            proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print('Subprocess timeout ' + str(TIMEOUT) + "s")
            killProcess(proc)
        except Exception as error:
            logging.error(f"Unexpected {error=}, {type(error)=}")


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

    logging.info("Initial sample file '" + FILE_CLASSES_INIT + "'")
    initSample = getInitSample()

    logging.info("Total number of classes: " + str(len(initSample)))
    logging.info("Random sample selection: " + str(RANDOM))
    logging.info("Sample size: " + str(SAMPLE_SIZE))
    logging.info("Executions/Class: " + str(EXECUTIONS_PER_CLASS))
    logging.info("Search budget: " + str(SEARCH_BUDGET) + "s")
    logging.info("Timeout: " + str(TIMEOUT) + "s")
    logging.info("Algorithm: " + ALGORITHM)

    if SAMPLE_SIZE > len(initSample):
        logging.error("sample size '" + str(SAMPLE_SIZE) + "' > init file length '" + str(len(initSample)) + "'")
        return

    # select sample
    sample = selectSample(initSample)

    # save backup
    createBackups(initSample, sample)

    # run tests
    for i in sample:
        project = initSample[i][0]
        clazz = initSample[i][1]
        runEvoSuite(project, clazz, i)


if __name__ == "__main__":
    main()
