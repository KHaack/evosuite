# -*- coding: utf-8 -*-
"""
    Purpose: Collect the experiment results from the experiment_runner
    Author: Kevin Haack
"""
import argparse
import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# filter
FILTER_MIN_EXECUTIONS = 9


def getStatisticFiles():
    """
    Determines all statistic files.
    Returns: A list of all files
    """
    pattern = os.path.join(PATH_WORKING_DIRECTORY, args.results, '**', '*.csv')
    logging.debug('use pattern: ' + pattern)
    return list(glob.iglob(pattern, recursive=True))


def evaluation(dataframe):
    """
    Creates a evaluation for the passed dataframe.
    :param dataframe: The dataframe for the evaluation
    """

    logging.info("CSV Directory:\t\t" + args.results)
    totalLength = len(dataframe.index)
    logging.info("Total tests:\t\t" + str(totalLength))

    dataframe = ex.clean(dataframe)
    dataframe = ex.addAdditionalColumns(dataframe)
    dataframe = ex.getMeasurings(dataframe, 0)
    dataframe = ex.filter(dataframe, FILTER_MIN_EXECUTIONS)

    # ##################################
    # print infos
    # ##################################
    logging.info("---------------------------------------------------------")
    logging.info("Tests for evaluation:\t" + str(len(dataframe.index)))
    logging.info("Java classes:\t\t" + str(len(dataframe.groupby('TARGET_CLASS'))))
    logging.info("Mean execution/class:\t" + str(dataframe.groupby('TARGET_CLASS').count().mean()[0]))
    logging.info("Branches max:\t\t" + str(dataframe['Total_Branches'].max()))
    logging.info("Branches min:\t\t" + str(dataframe['Total_Branches'].min()))
    logging.info("Gradient branches max:\t" + str(dataframe['Gradient_Branches'].max()))
    logging.info("Gradient branches min:\t" + str(dataframe['Gradient_Branches'].min()))
    logging.info("Lines max:\t\t" + str(dataframe['Lines'].max()))
    logging.info("Lines min:\t\t" + str(dataframe['Lines'].min()))

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at the end of the search")
    logging.info("----------------------------------------------------------------------------------------------------")

    printCorrelations('final-state', dataframe, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("----------------------------------------------------------------------------------------------------")

    percentReached = 1
    subset = ex.getMeasurings(dataframe, percentReached)
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')

    logging.info("----------------------------------------------------------")

    percentReached = 2
    subset = ex.getMeasurings(dataframe, percentReached)
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("-- Remove zero branches")
    logging.info("----------------------------------------------------------------------------------------------------")

    percentReached = 2
    subset = ex.getMeasurings(dataframe, percentReached)
    subset = subset[subset['Total_Branches'].gt(0)]
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')

    # ###############################################################
    # ###############################################################
    # ###############################################################
    # ex.createExport(subset[subset['_BranchRatio'].lt(0.2)])
    # foo_class_variance(dataframe)

    logging.info("----------------------------------------------------------------------------------------------------")

    percentReached = 2
    subset = ex.getMeasurings(dataframe, percentReached)

    logging.info("@20%: _BranchRatio <= 0.2 and _BranchRatio > 0.0")
    subset['predicted'] = subset['_BranchRatio'].le(0.2) & subset['_BranchRatio'].gt(0)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _Fitness >= 0.8 and _BranchRatio <= 0.2")
    subset['predicted'] = subset['_Fitness'].ge(0.8) & subset['_BranchRatio'].le(0.2)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _BranchRatio <= 0.2")
    subset['predicted'] = subset['_BranchRatio'].le(0.2)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _Fitness >= 0.8")
    subset['predicted'] = subset['_Fitness'].ge(0.8)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _BranchRatio <= 0.2 and not Branchless")
    subset['predicted'] = subset['_BranchRatio'].le(0.2) & ~subset['Branchless']
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: Branchless")
    subset['predicted'] = subset['Branchless']
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _BranchRatio <= 0.2 and Total_Branches > 0 and _Fitness >= 0.8")
    subset['predicted'] = subset['_BranchRatio'].le(0.2) & subset['Total_Branches'].gt(0) & subset['_Fitness'].ge(0.8)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20%: _NotCovGraRatio <= 0.1 and _Fitness >= 0.8")
    subset['predicted'] = subset['_NotCovGraRatio'].le(0.1) & subset['_Fitness'].ge(0.8)
    calculateF1(subset)
    logging.info("----------------------------")

    logging.info("@20% (without branchless): _Fitness == 0 or (_BranchRatio <= 0.2 and (_Fitness >= 0.2 and _NeutralityGen >= 0.2))")
    percentReached = 2
    subset = ex.getMeasurings(dataframe, percentReached)
    subset = subset[~subset['Branchless']]
    subset = subset[subset['_BranchRatio'].gt(0)]

    subset['predicted'] = subset['_GradientRatio'].lt(1)
    calculateF1(subset)
    logging.info("----------------------------")

    # ###############################################################
    percentReached = 2
    subset = ex.getMeasurings(dataframe, percentReached)
    #subset = subset[~subset['Branchless']]
    #subset = subset[subset['_BranchRatio'].gt(0)]

    ax = plt.axes(projection='3d')
    # GroundTruth, Branchless, CoverageClass, Coverage
    # _NotGradRatio, _GradientRatio, _BranchRatio, _NotCovGraRatio, _Fitness
    # _InfoContent, _NeutralityGen
    ax.scatter3D(subset['_BranchRatio'], subset['_Fitness'], subset['Coverage'], c=subset['GroundTruth'])
    ax.set_xlabel('_BranchRatio at 20%')
    ax.set_ylabel('_Fitness at 20%')
    ax.set_zlabel('Coverage at 20%')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    plt.show()


def foo_class_variance(dataframe):
    percentReached = 2
    dataframe = ex.getMeasurings(dataframe, percentReached)
    dataframe['matched'] = dataframe['_BranchRatio'].le(0.2)

    groups = dataframe.groupby('TARGET_CLASS').agg({
        'Coverage': 'var',
        '_Fitness': 'mean',
        '_BranchRatio': 'mean',
        'matched': 'mean'
    }).reset_index()

    ax = plt.axes(projection='3d')
    ax.scatter3D(groups['_BranchRatio'], groups['_Fitness'], groups['Coverage'], c=groups['matched'])
    ax.set_xlabel('_BranchRatio (mean) at 20%')
    ax.set_ylabel('_Fitness (mean) at 20%')
    ax.set_zlabel('End Coverage (variance)')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)


def calculateF1(dataframe):
    # A test result that correctly indicates the presence of a condition or characteristic
    dataframe['condition'] = ''
    dataframe.loc[(dataframe['predicted'].eq(True) & dataframe['GroundTruth'].eq(True)), 'condition'] = 'TP'
    # A test result which wrongly indicates that a particular condition or attribute is present
    dataframe.loc[(dataframe['predicted'].eq(True) & dataframe['GroundTruth'].eq(False)), 'condition'] = 'FP'
    # A test result which wrongly indicates that a particular condition or attribute is absent
    dataframe.loc[(dataframe['predicted'].eq(False) & dataframe['GroundTruth'].eq(True)), 'condition'] = 'FN'

    precision = 0.0
    recall = 0.0
    if len(dataframe[dataframe['condition'].eq('TP')].index) > 0:
        precision = len(dataframe[dataframe['condition'].eq('TP')].index) / (
                len(dataframe[dataframe['condition'].eq('TP')].index) + len(
            dataframe[dataframe['condition'].eq('FP')].index))
        recall = len(dataframe[dataframe['condition'].eq('TP')].index) / (
                len(dataframe[dataframe['condition'].eq('TP')].index) + len(
            dataframe[dataframe['condition'].eq('FN')].index))

    f1 = 2 * (precision * recall) / (precision + recall)

    logging.info(f'precision: {precision}')
    logging.info(f'recall: {recall}')
    logging.info(f'f_1: {f1}')


def printCorrelations(name, dataframe, referenceColumn, enablePlotting=False):
    """
    Calculate the correlation of the passed dataframe.
    :param name: the name of the passed data.
    :param dataframe: the dataframe.
    """
    logging.info(name + " (count: " + str(len(dataframe.index)) + ")")

    if len(dataframe.index) > 1:
        logging.info("\t\t\t\t\tpearson\t\t\tp-value\t\t\t|spearman\t\tp-value")
        printCorrelation(name, dataframe, '_Fitness', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_BranchRatio', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_GradientRatio', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_NotGradRatio', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_InfoContent', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_NeutralityVol', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_NeutralityGen', referenceColumn, enablePlotting)
        printCorrelation(name, dataframe, '_Generations', referenceColumn, enablePlotting)


def printCorrelation(name, dataframe, x, y, enablePlotting):
    """
    Get the correlation of the passed variables x and y.

    :param name: The name of that dataframe.
    :param dataframe: The dataframe.
    :param x: Variable x.
    :param y: Variable y.
    :return: None
    """
    pearson = stats.pearsonr(dataframe[x], dataframe[y])
    spearman = stats.spearmanr(dataframe[x], dataframe[y])
    logging.info(
        x + "-" + y + "\t\t" + str(pearson[0]) + "\t" + str(pearson[1]) + "\t|" + str(spearman[0]) + "\t" + str(
            spearman[1]))

    if enablePlotting:
        plot(name, dataframe, x, y)


def plot(name, dataframe, x, y):
    """
    Plot the passed variables x and y.

    :param name: The name of that dataframe.
    :param dataframe: The dataframe.
    :param x: Variable x.
    :param y: Variable y.
    :return: None
    """
    plotDir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_PLOT)

    if not os.path.exists(plotDir):
        os.mkdir(plotDir)

    title = name + "-" + x + "-" + y
    dataframe.plot.scatter(x=x, y=y, alpha=0.5, title=title)

    plotFile = os.path.join(plotDir, title + ".png")
    plt.savefig(plotFile)
    plt.close(plotFile)


def setupArgparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(description="Collect the experiment results from the experiment_runner",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-results", help="The directory of the results", type=ex.dir_path, required=True)

    return parser


def main():
    """
    Runs large scale experiment.
    """
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    files = getStatisticFiles()
    logging.info("search statistic files...")
    logging.info("found statistic files:\t" + str(len(files)))

    li = []
    for file in files:
        li.append(pd.read_csv(file, delimiter=','))

    dataframe = pd.concat(li, axis=0, ignore_index=True)

    logging.info("start evaluation...")
    evaluation(dataframe)


if __name__ == "__main__":
    parser = setupArgparse()
    args = parser.parse_args()
    main()
