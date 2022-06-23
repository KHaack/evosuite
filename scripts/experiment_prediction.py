# -*- coding: utf-8 -*-
"""
    Purpose: Collect the experiment results and make a prediction.
    Author: Kevin Haack
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from dtreeviz.trees import dtreeviz
from IPython.display import Image, display_svg, SVG

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# filter
FILTER_MIN_EXECUTIONS = 25
SCATTER_POINT_SIZE = 4

RANDOM_STATE = 42


def predict(title, dataframe, ground_truth_name, ground_truth, model, make_plots=False, print_tree=False):
    count_true = len(dataframe[dataframe[ground_truth]])
    count_false = len(dataframe[~dataframe[ground_truth]])

    logging.info(f'{ground_truth_name} (True): {count_true}')
    logging.info(f'{ground_truth_name} (False): {count_false}')

    balance = count_true / (count_true + count_false)
    logging.info(f'balancing: {balance}')

    if balance < 0.25 or balance > 0.75:
        logging.info('Up-sample minority class...')
        if balance > 0.75:
            majority = dataframe[dataframe[ground_truth]]
            minority = dataframe[~dataframe[ground_truth]]
        else:
            majority = dataframe[~dataframe[ground_truth]]
            minority = dataframe[dataframe[ground_truth]]

        # sample with replacement
        minority = resample(minority, replace=True, n_samples=len(majority.index), random_state=RANDOM_STATE)
        dataframe = pd.concat([majority, minority])

        logging.info(f'{ground_truth_name} (True): {len(dataframe[dataframe[ground_truth]])}')
        logging.info(f'{ground_truth_name} (False): {len(dataframe[~dataframe[ground_truth]])}')
        logging.info(f'balancing: {len(dataframe[dataframe[ground_truth]]) / len(dataframe)}')

    x_names = ['Branchless', '_GradientRatio', '_BranchRatio', '_Fitness', '_InfoContent', '_NeutralityGen']
    x = dataframe[x_names]
    y = dataframe[ground_truth].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    logging.info('fit train data...')
    model.fit(x_train, y_train)

    logging.info('predict test data...')
    y_prediction = model.predict(x_test)

    if print_tree:
        text_representation = tree.export_text(model, feature_names=list(x.columns))
        print(text_representation)

    if make_plots:
        # feature importances
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        plt.title(f'Feature Importance: {title}')
        plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
        plt.xticks(range(x_train.shape[1]), x.columns[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()

        # confusion matrix
        cm = confusion_matrix(y_test, y_prediction, labels=model.classes_, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title(f'Confusion matrix: {title}')
        plt.show()

        # dtreeviz
        viz = dtreeviz(model,
                       title=f'Decision tree - {ground_truth_name}',
                       x_data=x_train,
                       y_data=y_train,
                       feature_names=x_names,
                       class_names=['false', 'true'])
        viz.view()

    logging.info('get classification_report...')
    print(classification_report(y_test, y_prediction))


def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(description="Collect the experiment results and make a prediction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-results", help="The directory of the results", type=ex.dir_path, required=True)

    return parser


def main():
    path = os.path.join(PATH_WORKING_DIRECTORY, args.results)
    dataframe = ex.get_statistics(path)
    dataframe = ex.clean(dataframe)
    dataframe = ex.add_additional_columns(dataframe)
    dataframe = ex.filter_dataframe(dataframe, FILTER_MIN_EXECUTIONS)

    dataframe = ex.get_measurements(dataframe, -1)
    ex.print_result_infos(dataframe)

    logging.info("start prediction...")
    logging.info("@20%")
    dataframe = ex.get_measurements(dataframe, 2)

    model = tree.DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE, criterion="gini")

    # predict('All @20%', dataframe, 'High relative coverage', 'High relative coverage', model, make_plots=True)
    # predict('All @20%', dataframe, 'Well performing', 'Well performing', model, make_plots=True)
    predict('All @20%', dataframe, 'Low Coverage (std)', 'Low Coverage (std)', model, make_plots=True)


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
