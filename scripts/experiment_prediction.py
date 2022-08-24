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
from dtreeviz.trees import dtreeviz
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils import resample

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# filter
FILTER_MIN_EXECUTIONS = 25
SCATTER_POINT_SIZE = 4

RANDOM_STATE = 42


def is_leaf(inner_tree, index):
    """
    Check whether node is leaf node.
    https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
    :param inner_tree:
    :param index:
    :return:
    """
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    """
    Start pruning from the bottom - if we start from the top, we might miss nodes that become leaves during pruning.
    Do not use this directly - use prune_duplicate_leaves instead.
    https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
    :param inner_tree:
    :param decisions:
    :param index:
    :return:
    """
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))


def prune_duplicate_leaves(model):
    """
    Remove leaves if both.
    :param model:
    :return:
    """
    decisions = model.tree_.value.argmax(axis=2).flatten().tolist()
    prune_index(model.tree_, decisions)


def create_report(title, model, y_test, y_prediction, make_plots=False):
    if make_plots:
        # confusion matrix
        confusion_norm = confusion_matrix(y_test, y_prediction, labels=model.classes_, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_norm, display_labels=model.classes_)
        disp.plot()
        plt.title(f'Confusion matrix - {title}')
        plt.show()

    logging.info('get classification_report...')
    report = classification_report(y_test, y_prediction, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_prediction, labels=model.classes_).ravel()

    report['FPR'] = fp / (fp + tn)
    report['TPR'] = tp / (tp + fn)

    return report


def predict(title, dataframe, target, model, features, make_plots=False, print_tree=False, prune_tree=False,
            export_prediction=False, upsample=False):
    count_true = len(dataframe[dataframe[target]])
    count_false = len(dataframe[~dataframe[target]])

    logging.info(f'{target} (True): {count_true}')
    logging.info(f'{target} (False): {count_false}')

    balance = count_true / (count_true + count_false)
    logging.info(f'balancing: {balance}')

    if upsample:
        if balance < 0.25 or balance > 0.75:
            logging.info('Up-sample minority class...')
            if balance > 0.75:
                majority = dataframe[dataframe[target]]
                minority = dataframe[~dataframe[target]]
            else:
                majority = dataframe[~dataframe[target]]
                minority = dataframe[dataframe[target]]

            # sample with replacement
            minority = resample(minority, replace=True, n_samples=len(majority.index), random_state=RANDOM_STATE)
            dataframe = pd.concat([majority, minority])

            logging.info(f'{target} (True): {len(dataframe[dataframe[target]])}')
            logging.info(f'{target} (False): {len(dataframe[~dataframe[target]])}')
            logging.info(f'balancing: {len(dataframe[dataframe[target]]) / len(dataframe)}')

    x = dataframe[features]
    y = dataframe[target].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    logging.info('fit train data...')
    model.fit(x_train, y_train)

    if prune_tree:
        logging.info('prune tree...')
        prune_duplicate_leaves(model)

    logging.info('predict test data...')
    y_prediction = model.predict(x_test)

    if export_prediction:
        x_test['prediction'] = y_prediction
        export = pd.merge(dataframe, x_test, left_index=True, right_index=True)
        ex.create_export(export[export['prediction']])

    if print_tree:
        logging.info('print tree...')
        text_representation = tree.export_text(model, feature_names=list(x.columns))
        print(text_representation)

    if make_plots:
        logging.info('make plots...')
        # feature importances
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        plt.title(f'Feature Importance - {title}')
        plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
        plt.xticks(range(x_train.shape[1]), x.columns[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()

        # dtreeviz
        viz = dtreeviz(model,
                       title=f'Decision tree - {title}',
                       x_data=x_train,
                       y_data=y_train,
                       feature_names=features,
                       class_names=['false', 'true'])
        viz.view()

    return create_report(title, model, y_test, y_prediction, make_plots)


def compare_prediction(dataframe, targets, features, to_csv=False, upsample=True):
    """
    Compare the results of different predictions.
    :param targets:
    :param dataframe: The dataframe
    :return: None
    """
    rows = []
    criterion = ['gini', 'entropy', 'log_loss']

    for percentage in range(1, 4):
        dataframe = ex.get_measurements(dataframe, percentage)
        for c in criterion:
            for target in targets:
                for depth in range(1, 5):
                    logging.info(f"prediction of: {target} @ {percentage * 10}...")
                    model = tree.DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE, criterion=c,
                                                        class_weight='balanced')
                    report = predict(f"@{percentage * 10}%", dataframe, target, model, features,
                                     make_plots=False, print_tree=False, upsample=True)

                    row = {
                        'target': target,
                        'percentage': f"{percentage * 10}%",
                        'depth': depth,
                        'criterion': c,
                        'accuracy': report['accuracy'],
                        'TPR': report['TPR'],
                        'FPR': report['FPR'],
                        'true - precision': report['True']['precision'],
                        'true - recall': report['True']['recall'],
                        'true - f1': report['True']['f1-score'],
                        'false - precision': report['False']['precision'],
                        'false - recall': report['False']['recall'],
                        'false - f1': report['False']['f1-score']
                    }
                    rows.append(row)

    logging.info(f"created {len(rows)} decision trees.")

    results_best = pd.DataFrame()
    results_all = pd.DataFrame(rows)
    for target in targets:
        results_best = pd.concat(
            [results_best,
             results_all[results_all['target'].eq(target)].sort_values(by=['FPR'], ascending=True).head(5)])

    results_best = results_best.round(2)
    results_best = results_best.sort_values(by=['FPR'], ascending=True)

    if to_csv:
        results_best[['target', 'percentage', 'depth', 'criterion', 'accuracy', 'TPR', 'FPR']].to_csv('predictions.csv')

    print(results_best)


def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(description="Collect the experiment results and make a prediction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-results", help="The directory of the results", type=ex.check_dir_path, required=True)
    parser.add_argument("-evaluation", help="The directory of the evaluation results", type=ex.check_dir_path, required=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-single_prediction", help="Make the implemented prediction", action='store_true')
    group.add_argument("-compare_predictions", help="Compares all predictions", action='store_true')

    return parser


def add_additional_coverage(dataframe, column_name, folder):
    other = ex.get_statistics(folder)

    groups = other.groupby(['TARGET_CLASS']).agg({
        'Coverage': 'median'
    })
    groups.reset_index(inplace=True)

    merged = dataframe.merge(groups, how='inner', left_on='TARGET_CLASS', right_on='TARGET_CLASS')
    merged.rename({
        'Coverage': column_name
    }, inplace=True, axis=1)

    return merged


def get_dataframe(folder):
    path = os.path.join(PATH_WORKING_DIRECTORY, folder)
    dataframe = ex.get_statistics(path)
    dataframe = ex.clean(dataframe)
    dataframe = ex.add_additional_columns(dataframe)
    dataframe = ex.filter_dataframe(dataframe, FILTER_MIN_EXECUTIONS)

    return ex.get_measurements(dataframe, -1)


def main():
    dataframe = get_dataframe(args.results)
    dataframe = add_additional_coverage(dataframe, 'EndCoverage POP125', 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\33 PC with POP 125')
    ex.print_result_infos(dataframe)

    dataframe['HIGH_STDEV_and_RELATIVE_LOW_COVERAGE'] = dataframe['RELATIVE_LOW_COVERAGE'] & dataframe['HIGH_STDEV']
    dataframe['WITH_STDEV_and_RELATIVE_LOW_COVERAGE'] = dataframe['RELATIVE_LOW_COVERAGE'] & dataframe['WITH_STDEV']
    dataframe['HIGH_STDEV_and_LOW_END_COVERAGE'] = dataframe['LOW_END_COVERAGE'] & dataframe['HIGH_STDEV']
    dataframe['HIGHER_WITH_POP125'] = dataframe['EndCoverage'].lt(dataframe['EndCoverage POP125'])
    dataframe['HIGHER_WITH_POP125_and_RELATIVE_LOW_COVERAGE'] = dataframe['RELATIVE_LOW_COVERAGE'] & dataframe['HIGHER_WITH_POP125']

    targets = ['RELATIVE_LOW_COVERAGE',
               'LOW_END_COVERAGE',
               'HIGH_STDEV',
               'WITH_STDEV',
               'HIGH_STDEV_and_RELATIVE_LOW_COVERAGE',
               'WITH_STDEV_and_RELATIVE_LOW_COVERAGE',
               'HIGH_STDEV_and_LOW_END_COVERAGE',
              'HIGHER_WITH_POP125',
              'HIGHER_WITH_POP125_and_RELATIVE_LOW_COVERAGE'
               ]
    features = ['Branchless', 'GradientRatio', 'BranchRatio', 'Fitness', 'InformationContent', 'NeutralityRatio']

    if args.compare_predictions:
        compare_prediction(dataframe, targets, features, to_csv=True, upsample=True)
    if args.single_prediction:
        percentage = 3
        dataframe = ex.get_measurements(dataframe, percentage)
        model = tree.DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE, criterion="gini",
                                            class_weight='balanced')
        target = 'HIGH_STDEV_and_RELATIVE_LOW_COVERAGE'
        predict("Decision Tree applied on $S_1$", dataframe, target, model,
                features,
                make_plots=True, prune_tree=True, export_prediction=True, upsample=True)

        if args.evaluation is not None:
            logging.info(f"evaluation results...")

            dataframe_eval = get_dataframe(args.evaluation)
            dataframe_eval['HIGH_STDEV_and_RELATIVE_LOW_COVERAGE'] = dataframe_eval['RELATIVE_LOW_COVERAGE'] & dataframe_eval['HIGH_STDEV']
            x = dataframe_eval[features]
            y = dataframe_eval[target].values
            y_prediction = model.predict(x)
            report = create_report('Decision Tree applied on $S_2$', model, y, y_prediction, make_plots=True)

            print(report)


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
