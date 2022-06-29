# -*- coding: utf-8 -*-
"""
    Purpose: Collect the experiment results from the experiment_runner
    Author: Kevin Haack
"""
import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np

import experiment_lib as ex

pd.options.mode.chained_assignment = None

# files and folders
DIRECTORY_PLOT = "plots"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# filter
FILTER_MIN_EXECUTIONS = 25
SCATTER_POINT_SIZE = 4


def euclidean_distance(dataframe1, dataframe2, columns):
    """
    Calculates the euclidean distance between the two passed dataframes.
    :param dataframe1: Dataframe 1
    :param dataframe2: Dataframe 2
    :param columns: The columns for the euclidean distance
    :return: the euclidean distance
    """
    return np.linalg.norm(dataframe1[columns].values - dataframe2[columns].values, axis=1)


def foo_general_infos(original, dataframe):
    # percentage histogram
    ax = original.hist(column='PercentageReached', bins=10)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Percentage reached")
    plt.title('Histogram - Percentage reached')
    plt.tight_layout()
    plt.show()

    # coverage histogram
    ax = dataframe.hist(column='Coverage', bins=20)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Coverage")
    plt.title('Histogram - Coverage')
    plt.tight_layout()
    plt.show()

    # generations histogram
    rows = []
    for percentage in range(1, 11):
        subset = ex.get_measurements(dataframe, percentage)
        row = {
            'x': percentage - 1,
            'x-label': f'{percentage * 10}%',
            'mean': np.mean(subset['_Generations']),
            'median': np.median(subset['_Generations'])
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    ax = result.plot(kind='line', x='x', grid=True)
    ax.set_ylabel("Generations")
    ax.set_xlabel("Percentage")
    plt.title('Generations')
    plt.tight_layout()
    plt.xticks(np.arange(0, len(result['x'])), labels=result['x-label'])
    plt.show()


def foo_percentage_dif(dataframe):
    rows = []
    for i in range(2, 11):
        first = i - 1
        second = i

        subset10 = ex.get_measurements(dataframe, first)
        subset20 = ex.get_measurements(dataframe, second)
        subset10 = subset10[subset10['PercentageReached'].ge(second * 10)]

        subset10['Branchless'] = subset10['Branchless'].astype(float)
        subset20['Branchless'] = subset20['Branchless'].astype(float)

        columns = ['Branchless', '_GradientRatio', '_BranchRatio', '_Fitness', '_InfoContent', '_NeutralityGen']
        distance = euclidean_distance(subset10, subset20, columns)

        row = {
            'x': i - 2,
            'x-label': f"{first}0-{second}0%",
            'mean': np.mean(distance),
            'median': np.median(distance)
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    ax = result.plot(kind='line', x='x', rot=10, grid=True)
    ax.set_ylabel("Euclidean distance")
    ax.set_xlabel("Percentages")
    plt.title('Aggregated Euclidean distances')
    plt.tight_layout()
    plt.xticks(np.arange(0, len(result['x'])), labels=result['x-label'])
    plt.show()


def foo_correlation(dataframe):
    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at the end of the search")
    logging.info("----------------------------------------------------------------------------------------------------")

    print_correlations('final-state', dataframe, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("----------------------------------------------------------------------------------------------------")

    percent_reached = 1
    dataframe = ex.get_measurements(dataframe, percent_reached)
    print_correlations(str(percent_reached * 10) + 'p-time', dataframe, 'Coverage')

    logging.info("----------------------------------------------------------")

    percent_reached = 2
    dataframe = ex.get_measurements(dataframe, percent_reached)
    print_correlations(str(percent_reached * 10) + 'p-time', dataframe, 'Coverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("-- Remove zero branches")
    logging.info("----------------------------------------------------------------------------------------------------")

    percent_reached = 2
    dataframe = ex.get_measurements(dataframe, percent_reached)
    subset = dataframe[dataframe['Total_Branches'].gt(0)]
    print_correlations(str(percent_reached * 10) + 'p-time', subset, 'Coverage')


def foo_std(dataframe):
    groups = dataframe.groupby('TARGET_CLASS').agg({
        'Coverage': ['var', 'std', 'min', 'max', 'median'],
        'PERFORMS_BAD': 'mean',
        'Branchless': 'mean',
        '_GradientRatio': 'mean',
        '_BranchRatio': 'mean',
        '_NotGradRatio': 'mean',
        '_InfoContent': 'mean',
        '_NeutralityGen': 'mean',
        '_Fitness': 'mean'
    }).reset_index()

    groups[('Coverage', 'spread')] = groups[('Coverage', 'max')] - groups[('Coverage', 'min')]
    groups.sort_values(('Coverage', 'std'), inplace=True, ascending=False)

    fig, axs = plt.subplots(2, 3)
    draw_2d(axs[0, 0], fig, groups, 'GradientRatio at 20% (mean)', 'Coverage (std)', '_GradientRatio', ('Coverage', 'std'), title='a)')
    draw_2d(axs[0, 1], fig, groups, 'BranchRatio at 20% (mean)', 'Coverage (std)', '_BranchRatio', ('Coverage', 'std'), title='b)')
    draw_2d(axs[0, 2], fig, groups, 'NotGradRatio at 20% (mean)', 'Coverage (std)', '_NotGradRatio', ('Coverage', 'std'), title='c)')
    draw_2d(axs[1, 0], fig, groups, 'IC at 20% (mean)', 'Coverage (std)', '_InfoContent', ('Coverage', 'std'), title='d)')
    draw_2d(axs[1, 1], fig, groups, 'NV/Gen at 20% (mean)', 'Coverage (std)', '_NeutralityGen', ('Coverage', 'std'), title='e)')
    draw_2d(axs[1, 2], fig, groups, 'Fitness at 20% (mean)', 'Coverage (std)', '_Fitness', ('Coverage', 'std'), title='f)')
    plt.tight_layout()
    plt.show()


def foo_coverage(dataframe):
    fig, axs = plt.subplots(2, 3)
    draw_2d(axs[0, 0], fig, dataframe, 'GradientRatio at 20%', 'Coverage', '_GradientRatio', 'Coverage', title='a)')
    draw_2d(axs[0, 1], fig, dataframe, 'BranchRatio at 20%', 'Coverage', '_BranchRatio', 'Coverage', title='b)')
    draw_2d(axs[0, 2], fig, dataframe, 'NotGradRatio at 20%', 'Coverage', '_NotGradRatio', 'Coverage', title='c)')
    draw_2d(axs[1, 0], fig, dataframe, 'IC at 20%', 'Coverage', '_InfoContent', 'Coverage', title='d)')
    draw_2d(axs[1, 1], fig, dataframe, 'NV/Gen at 20%', 'Coverage', '_NeutralityGen', 'Coverage', title='e)')
    draw_2d(axs[1, 2], fig, dataframe, 'Fitness at 20%', 'Coverage', '_Fitness', 'Coverage', title='f)')
    plt.tight_layout()
    plt.show()


def draw_2d(ax, fig, dataframe, x_name, y_name, x, y, color=None, color_name=None, title=None):
    """
    Draw a 2d scatter plot.
    :param fig: The figure
    :param ax: The axis
    :param dataframe: The dataframe.
    :param x_name: The display name of the x column
    :param y_name: The display name of the y column
    :param color_name: The display name of the color column
    :param x: The name of the x column
    :param y: The name of the y column
    :param color: The name of the color column
    :return: None
    """
    if color is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        colormap = plt.cm.viridis

        ax.scatter(dataframe[x], dataframe[y], s=SCATTER_POINT_SIZE, c=dataframe[color].values, norm=norm, cmap=colormap)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        fig.colorbar(sm, label=color_name)
    else:
        ax.scatter(dataframe[x], dataframe[y], s=SCATTER_POINT_SIZE)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    if title is None:
        ax.set_title(f'{x_name} - {y_name}')
    else:
        ax.set_title(f'{title} {x_name} - {y_name}')

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_3d(dataframe, x_name, y_name, z_name, color_name, x, y, z, color):
    """
    Draw a 3d scatter plot.
    :param dataframe: The dataframe.
    :param x_name: The display name of the x column
    :param y_name: The display name of the y column
    :param z_name: The display name of the z column
    :param color_name: The display name of the color column
    :param x: The name of the x column
    :param y: The name of the y column
    :param z: The name of the z column
    :param color: The name of the color column
    :return: None
    """
    ax = plt.axes(projection='3d')
    fig = plt.gcf()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    colormap = plt.cm.viridis

    ax.scatter3D(dataframe[x], dataframe[y], dataframe[z], s=SCATTER_POINT_SIZE, c=dataframe[color])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    ax.set_xlim3d(0, 1.01)
    ax.set_ylim3d(0, 1.01)
    ax.set_zlim3d(0, 1.01)

    plt.title(f'{x_name} - {y_name} - {z_name}')

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, label=color_name)


def print_correlations(name, dataframe, referenceColumn, enablePlotting=False):
    """
    Calculate the correlation of the passed dataframe.
    :param name: the name of the passed data.
    :param dataframe: the dataframe.
    """
    logging.info(name + " (count: " + str(len(dataframe.index)) + ")")

    if len(dataframe.index) > 1:
        logging.info("\t\t\t\t\tpearson\t\t\tp-value\t\t\t|spearman\t\tp-value")
        print_correlation(name, dataframe, '_Fitness', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_BranchRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_GradientRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_NotGradRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_InfoContent', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_NeutralityVol', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_NeutralityGen', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, '_Generations', referenceColumn, enablePlotting)


def print_correlation(name, dataframe, x, y, enablePlotting):
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
    plot_dir = os.path.join(PATH_WORKING_DIRECTORY, DIRECTORY_PLOT)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    title = name + "-" + x + "-" + y
    dataframe.plot.scatter(x=x, y=y, alpha=0.5, title=title)

    plot_file = os.path.join(plot_dir, title + ".png")
    plt.savefig(plot_file)
    plt.close(plot_file)


def setup_argparse():
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
    path = os.path.join(PATH_WORKING_DIRECTORY, args.results)
    original = ex.get_statistics(path)
    original = ex.clean(original)
    original = ex.add_additional_columns(original)
    dataframe = ex.filter_dataframe(original, FILTER_MIN_EXECUTIONS)

    dataframe = ex.get_measurements(dataframe, -1)
    ex.print_result_infos(dataframe)

    logging.info("start evaluation...")

    foo_percentage_dif(dataframe)
    # foo_general_infos(original, dataframe)

    logging.info("@20%")
    dataframe = ex.get_measurements( dataframe, 2)

    # foo_correlation(dataframe)
    # foo_std(dataframe)
    # foo_coverage(dataframe)

    # ###################################################################

    # PERFORMS_BAD, HIGH_STDEV, Branchless, Coverage
    # _GradientRatio, _BranchRatio, _NotGradRatio
    # _InfoContent, _NeutralityGen
    # _Fitness
    # draw_3d(subset, 'BranchRatio at 20%', '_Fitness', 'Coverage', 'PERFORMS_BAD', '_BranchRatio', '_Fitness', 'Coverage', 'Well performing')
    # plt.show()


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
