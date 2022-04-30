# -*- coding: utf-8 -*-
"""
    Purpose: Run large scale experiments with EvoSuite
    Author: Kevin Haack (based on the batch script from Mitchell Olsthoorn)
"""
import glob
import logging
import os
import re
import subprocess
import sys

DIRECTORY_CORPUS = "SF110-20130704"
DIRECTORY_REPORTS = "reports"
DIRECTORY_TESTS = "tests"
DIRECTORY_LOGS = "logs"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
PATH_EVOSUITE = "C:\\Users\\kha\\repos\\evosuite\\master\\target\\evosuite-master-1.2.1-SNAPSHOT.jar"
FILE_CLASSES = "classes.txt"
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
EXECUTIONS_PER_CLASS = 1


def getProjectClassPath(project):
    """
    Determines the classpath based on the project and outputs this.
    Expects the following file structure: projects/<project>/<jars>
    Returns: Colon seperated class path
    """
    projectPath = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, project)
    logging.debug("create projectClassPath for folder '" + projectPath + "'")

    jarList = glob.iglob(os.path.join(projectPath, '**/*.jar'), recursive=True)
    classPath = "";
    for jar in jarList:
        classPath = classPath + jar + ";"

    return classPath


def runEvoSuiteSingle(project, clazz):
    """
    Runs a single execution of EvoSuite for the passed class.
    """
    folderPath = os.path.join(PATH_WORKING_DIRECTORY, project)
    logging.info(folderPath);


def runEvoSuite(project, clazz, numberOfClasses, numberCurrentClass):
    """
    Runs multiple executions of EvoSuite for the passed class.
    """
    projectClassPath = getProjectClassPath(project);

    for i in range(0, EXECUTIONS_PER_CLASS):
        logging.info("Class (" + str(numberCurrentClass) + " / " + str(numberOfClasses) + ") Execution (" + str(
            i + 1) + " / " + str(
            EXECUTIONS_PER_CLASS) + "): Running default configuration for class (" + clazz + ") in project (" + project + ") with random seed.")

        # output directories
        pathClassDir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, project, clazz)

        pathReport = os.path.join(pathClassDir, DIRECTORY_REPORTS, str(i))
        pathTest = os.path.join(pathClassDir, DIRECTORY_TESTS, str(i))
        pathLog = os.path.join(pathClassDir, DIRECTORY_LOGS)
        pathLogFile = os.path.join(pathClassDir, DIRECTORY_LOGS, "log_" + str(i) + ".txt")

        # create directories
        if not os.path.exists(pathClassDir):
            os.mkdir(pathClassDir)
        if not os.path.exists(pathLog):
            os.mkdir(pathLog)

        # build evoSuite parameters
        parameter = []
        parameter.append('java')
        parameter.append('-Xmx4G')
        parameter.append('-jar')
        parameter.append(PATH_EVOSUITE)
        parameter.append('-class')
        parameter.append(clazz)
        parameter.append('-projectCP')
        parameter.append(projectClassPath)

        parameter.append('-Dreport_dir=' + pathReport)
        parameter.append('-Dtest_dir=' + pathTest)
        parameter.append('-Dshow_progress=false')
        parameter.append('-Dplot=false')
        parameter.append('-Dclient_on_thread=false')

        parameter.append('-criterion')
        parameter.append('branch')
        parameter.append('-Denable_fitness_history=true')
        parameter.append('-Denable_landscape_analysis=true')
        parameter.append('-Dnew_statistics=true')
        parameter.append(
            '-Doutput_variables=Algorithm,TARGET_CLASS,Generations,criterion,Coverage,BranchCoverage,Total_Goals,Covered_Goals,NeutralityVolume,InformationContent')

        # start process
        output = open(pathLogFile, "w")
        subprocess.call(parameter, stdout=output, stderr=subprocess.STDOUT)
        output.close()


def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    pathFull = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_CORPUS, FILE_CLASSES)
    logging.info("open classes file '" + pathFull + "'")

    numberOfClasses = sum(1 for line in open(pathFull, "r"))
    logging.info("Total number of classes: " + str(numberOfClasses))

    classes = open(pathFull, "r")
    numberCurrentClass = 1
    for row in classes:
        parts = re.split('\t', row)
        project = parts[0]
        clazz = parts[1].replace('\n', '')

        runEvoSuite(project, clazz, numberOfClasses, numberCurrentClass)
        numberCurrentClass = numberCurrentClass + 1
        return


if __name__ == "__main__":
    main()
