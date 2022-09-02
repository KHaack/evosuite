# -*- coding: utf-8 -*-
"""
    Purpose: Compares the experiment results from two run of the experiment_runner
    Author: Kevin Haack
"""
import argparse
import logging
import pandas as pd
import scipy.stats as stats
import numpy as np

import experiment_lib as ex
import vargha_delaney as vd

pd.options.mode.chained_assignment = None

# filter
FILTER_MIN_EXECUTIONS = 25
SCATTER_POINT_SIZE = 4

def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    argument_parser = argparse.ArgumentParser(
        description="Compares the experiment results from two runs (a & b) of the the experiment_runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument("-a", help="The directory path of the run a", type=ex.check_dir_path, required=True)
    argument_parser.add_argument("-b", help="The directory path of the run b", type=ex.check_dir_path, required=True)
    argument_parser.add_argument("-export", help="export the classes", action='store_true')

    group = argument_parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-only_pc", help="compares only classes with pc=yes", action='store_true')
    group.add_argument("-only_not_pc", help="compares only classes with pc=no", action='store_true')

    return argument_parser


def main():
    """
    Runs large scale experiment.
    """
    dataframe_a = ex.get_statistics(args.a)
    dataframe_a['set'] = 'A'
    dataframe_a = ex.clean(dataframe_a)
    dataframe_a = ex.add_additional_columns(dataframe_a)
    dataframe_a = ex.get_measurements(dataframe_a, 3)

    dataframe_b = ex.get_statistics(args.b)
    dataframe_b['set'] = 'B'
    dataframe_b = ex.clean(dataframe_b)
    dataframe_b = ex.add_additional_columns(dataframe_b)
    dataframe_b = ex.get_measurements(dataframe_b, 3)

    logging.info("Total tests in A:\t\t" + str(len(dataframe_a.index)))
    logging.info("Total tests in B:\t\t" + str(len(dataframe_b.index)))
    logging.info("Total tests in B with PC:\t" + str(len(dataframe_b[dataframe_b['_ParameterControlled'].eq("yes")])))
    logging.info("Total tests in B without PC:\t" + str(len(dataframe_b[dataframe_b['_ParameterControlled'].eq("no")])))

    if args.only_pc:
        target_classes = np.unique(dataframe_b[dataframe_b["_ParameterControlled"].eq("yes")]['TARGET_CLASS'].values)
    elif args.only_not_pc:
        target_classes = np.unique(dataframe_b[dataframe_b["_ParameterControlled"].eq("no")]['TARGET_CLASS'].values)
    else:
        target_classes = np.unique(dataframe_b['TARGET_CLASS'].values)

    logging.info("Java Classes for test:\t\t" + str(len(target_classes)))

    significance_count = {
        'up': 0,
        'down': 0
    }

    rows = []
    for target_class in target_classes:
        a = dataframe_a[dataframe_a['TARGET_CLASS'].eq(target_class)]
        b = dataframe_b[dataframe_b['TARGET_CLASS'].eq(target_class)]

        if len(a) > 0 and len(b) > 0:
            mann = stats.mannwhitneyu(a['EndCoverage'], b['EndCoverage'], alternative='two-sided')
            delta, interpretation = vd.VD_A(a['EndCoverage'], b['EndCoverage'])

            if mann.pvalue <= 0.05 and (interpretation == 'large' or interpretation == 'medium'):
                if b['EndCoverage'].mean() > a['EndCoverage'].mean():
                    up = '$nearrow$'
                    significance_count['up'] = significance_count['up'] + 1
                else:
                    up = '$searrow$'
                    significance_count['down'] = significance_count['down'] + 1
            else:
                up = ''

            rows.append({
                'CUT': ex.short_java_class(target_class),
                'a mean': round(a['EndCoverage'].mean(), 3),
                'a std': round(a['EndCoverage'].std(), 3),
                'b mean': round(b['EndCoverage'].mean(), 3),
                'b std': round(b['EndCoverage'].std(), 3),
                '$\\hat{A}_{12}$': delta,
                'p-value': round(mann.pvalue, 3),
                '': up,
                'interpretation': interpretation
            })

    logging.info("significant up:\t\t\t" + str(significance_count['up']))
    logging.info("significant down:\t\t" + str(significance_count['down']))

    if len(rows) > 0:
        df = pd.DataFrame(rows)

        df = df.set_index('CUT')
        df = df.sort_values('p-value', ascending=True)

        print(df)

        if args.export:
            df.to_csv('export.csv')


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
