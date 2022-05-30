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
import numpy as np
from sklearn import tree

import experiment_lib as ex

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# filter
FILTER_MIN_EXECUTIONS = 20


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
    dataframe = ex.add_additional_columns(dataframe)
    dataframe = ex.get_measurements(dataframe, 0)
    dataframe = ex.filter_dataframe(dataframe, FILTER_MIN_EXECUTIONS)

    # ##################################
    # print infos
    # ##################################
    logging.info("---------------------------------------------------------")
    logging.info(f"Tests for evaluation:\t{str(len(dataframe.index))}")
    logging.info(f"Java classes:\t\t{str(len(dataframe.groupby('TARGET_CLASS')))}")
    logging.info(f"Mean execution/class:\t{str(dataframe.groupby('TARGET_CLASS').count().mean()[0])}")
    logging.info(f"Branches max:\t\t{str(dataframe['Total_Branches'].max())}")
    logging.info(f"Branches min:\t\t{str(dataframe['Total_Branches'].min())}")
    logging.info(f"Gradient branches max:\t{str(dataframe['Gradient_Branches'].max())}")
    logging.info(f"Gradient branches min:\t{str(dataframe['Gradient_Branches'].min())}")
    logging.info(f"Lines max:\t\t{str(dataframe['Lines'].max())}")
    logging.info(f"Lines min:\t\t{str(dataframe['Lines'].min())}")
    logging.info(f'GroundTruth (True):\t{len(dataframe[dataframe["GroundTruth"]])}')
    logging.info(f'GroundTruth (False):\t{len(dataframe[~dataframe["GroundTruth"]])}')
    logging.info(f'Branchless:\t\t{len(dataframe[dataframe["Branchless"]])}')
    logging.info("---------------------------------------------------------")

    # ex.createExport(subset[subset['_BranchRatio'].lt(0.2)])
    # foo_correlation(dataframe)
    # foo_class_variance(dataframe)

    # foo_predict('RandomForest', dataframe, RandomForestClassifier(n_estimators=100, random_state=42, max_features="sqrt"))

    # ###################################################################
    logging.info("Predict @20%: all")
    subset = ex.get_measurements(dataframe, 2)

    foo_predict('All @20%', subset, tree.DecisionTreeClassifier(max_depth=5, random_state=42), True)
    # foo_common_plots('all', subset, '_BranchRatio', '_Fitness', 'Coverage')


def foo_predict(title, dataframe, model, make_plots=False):
    logging.info(f'GroundTruth (True): {len(dataframe[dataframe["GroundTruth"]])}')
    logging.info(f'GroundTruth (False): {len(dataframe[~dataframe["GroundTruth"]])}')
    logging.info(f'balancing: {len(dataframe[dataframe["GroundTruth"]]) / len(dataframe[~dataframe["GroundTruth"]])}')

    x = dataframe[['Branchless', '_GradientRatio', '_BranchRatio', '_Fitness', '_InfoContent', '_NeutralityGen']]
    y = dataframe['GroundTruth'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    logging.info('fit train data...')
    model.fit(x_train, y_train)

    logging.info('predict test data...')
    prediction = model.predict(x_test)

    logging.info('get classification_report...')
    print(classification_report(y_test, prediction))

    if make_plots:
        # plot tree
        # tree.plot_tree(model, feature_names=x.columns, rounded=True, filled=True, class_names=['good', 'bad'])
        # plt.show()
        text_representation = tree.export_text(model, feature_names=list(x.columns))
        print(text_representation)

        # feature importances
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        plt.title(f'Feature Importance: {title}')
        plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
        plt.xticks(range(x_train.shape[1]), x.columns[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()

        # confusion matrix
        cm = confusion_matrix(y_test, prediction, labels=model.classes_, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title(f'Confusion matrix: {title}')
        plt.show()

    logging.info('predict test data...')
    logging.info(f'accuracy_score: {accuracy_score(y_test, prediction)}')


def foo_correlation(dataframe):
    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at the end of the search")
    logging.info("----------------------------------------------------------------------------------------------------")

    printCorrelations('final-state', dataframe, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("----------------------------------------------------------------------------------------------------")

    percentReached = 1
    subset = ex.get_measurements(dataframe, percentReached)
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')

    logging.info("----------------------------------------------------------")

    percentReached = 2
    subset = ex.get_measurements(dataframe, percentReached)
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("-- Remove zero branches")
    logging.info("----------------------------------------------------------------------------------------------------")

    percentReached = 2
    subset = ex.get_measurements(dataframe, percentReached)
    subset = subset[subset['Total_Branches'].gt(0)]
    printCorrelations(str(percentReached * 10) + 'p-time', subset, 'Coverage')


def foo_common_plots(title, dataframe, x, y, z):
    plt.title(f"Coverage histogram: {title}")
    plt.hist(dataframe['Coverage'])
    plt.plot()
    plt.show()

    # ################################
    ax = plt.axes(projection='3d')
    # GroundTruth, Branchless, Coverage
    # _GradientRatio, _BranchRatio, _NotGradRatio
    # _InfoContent, _NeutralityGen
    # _Fitness
    ax.scatter3D(dataframe[x], dataframe[y], dataframe[z], c=dataframe['GroundTruth'])
    ax.set_xlabel(f'{x} at 20%')
    ax.set_ylabel(f'{y} at 20%')
    ax.set_zlabel(f'{z} at 20%')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    plt.title(title)
    plt.show()


def foo_class_variance(dataframe):
    percentReached = 2
    dataframe = ex.get_measurements(dataframe, percentReached)
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
