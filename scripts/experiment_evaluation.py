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
import numpy as np
import pandas as pd
import scipy.stats as stats

import experiment_lib as ex
import seaborn as sns

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


def foo_generations(original):
    # generations
    rows = []
    for percentage in range(1, 11):
        subset = ex.get_measurements(original, percentage)
        row = {
            'x': percentage - 1,
            'x-label': f'{percentage * 10}%',
            'mean': np.mean(subset['Generations']),
            'median': np.mean(subset['Generations'])
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    ax = result.plot(kind='line', x='x', grid=True)
    ax.set_ylabel("Generations")
    ax.set_xlabel("Percentage")

    plt.title('Generations')
    plt.tight_layout()
    plt.xticks(np.arange(0, len(result['x'])), labels=result['x-label'])
    plt.legend()
    plt.show()

    # generations with PC
    folders = [
        {
            'label': 'Rank bias = 2.0',
            'color': 'darkorange',
            'linestyle': 'solid',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\34 PC with BIAS 2.0\\'
        },
        {
            'label': 'Rank bias = 1.01',
            'color': 'darkorange',
            'linestyle': 'dashed',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\35 PC with BIAS 1.01\\'
        },
        {
            'label': 'Population = 125',
            'color': 'darkblue',
            'linestyle': 'solid',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\37 PC POP 125\\'
        },
        {
            'label': 'Population = 25',
            'color': 'darkblue',
            'linestyle': 'dashed',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\38 PC POP 25\\'
        },
        {
            'label': 'Test insert = 0.2',
            'color': 'red',
            'linestyle': 'solid',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\39 PC PTI 0.2\\'
        },
        {
            'label': 'Test insert = 0.0',
            'color': 'red',
            'linestyle': 'dashed',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\45 PC PTI 0\\'
        },
        {
            'label': 'Crossover = 1.0',
            'color': 'darkgreen',
            'linestyle': 'solid',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\43 PC CO 1.0\\'
        },
        {
            'label': 'Crossover = 0.0',
            'color': 'darkgreen',
            'linestyle': 'dashed',
            'folder': 'C:\\Users\\kha\\Desktop\\Benchmark\\results\\41 PC CO 0.0\\'
        }
    ]

    rows = []
    for percentage in range(1, 11):
        subset = ex.get_measurements(original, percentage)
        row = {
            'x': percentage - 1,
            'x-label': f'{percentage * 10}%',
            'generations': np.mean(subset['Generations']),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    plt.plot(result['x'], result['generations'], label='default', color='black', linestyle='solid')

    for folder in folders:
        dataframe = ex.get_statistics(folder['folder'])
        dataframe = ex.clean(dataframe)
        dataframe = ex.add_additional_columns(dataframe)
        dataframe = ex.filter_dataframe(dataframe, FILTER_MIN_EXECUTIONS)

        rows = []
        for percentage in range(1, 11):
            subset = ex.get_measurements(dataframe, percentage)
            yes = subset[subset["_ParameterControlled"].eq("yes")]

            row = {
                'x': percentage - 1,
                'x-label': f'{percentage * 10}%',
                'generations': np.mean(yes['Generations'])
            }
            rows.append(row)

        result = pd.DataFrame(rows)
        plt.plot(result['x'], result['generations'], label=folder['label'], color=folder['color'],
                 linestyle=folder['linestyle'], alpha=0.4)

    ax = plt.gca()
    ax.set_ylabel("Generations (mean)")
    ax.set_xlabel("Percentage")
    ax.text(2.1, 600, 'POC', alpha=0.6)

    plt.title(f'Generations - default vs. parameter')
    plt.tight_layout()
    plt.grid(alpha=0.2)
    plt.axvline(2, 0, 1600, linestyle='dotted', alpha=0.6)
    plt.legend()
    plt.xticks(np.arange(0, len(result['x'])), labels=result['x-label'])
    plt.show()


def foo_general_infos(original, dataframe):
    # HIGH_STDEV histogram
    groups = dataframe.groupby('TARGET_CLASS').agg({
        'Coverage (std)': ['max']
    }).reset_index()

    ax = groups.hist(column=('Coverage (std)', 'max'), bins=10)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Coverage (std)")
    plt.title('Histogram - coverage (std)')
    plt.tight_layout()
    plt.show()

    # percentage histogram
    ax = original.hist(column='Total_Branches', bins=10)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Branches")
    plt.title('Histogram - Branches $S_2$')
    plt.tight_layout()
    plt.show()

    # percentage histogram
    ax = original.hist(column='PercentageReached', bins=10)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Percentage of search budget reached")
    plt.title('Histogram - used search budget $S_2$')
    plt.tight_layout()
    plt.show()

    # coverage histogram
    ax = dataframe.hist(column='EndCoverage', bins=20)
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_xlabel("Coverage")
    plt.title('Histogram - Coverage $S_2$')
    plt.tight_layout()
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

        columns = ['Branchless', 'GradientRatio', 'BranchRatio', 'Fitness', 'InformationContent', 'NeutralityRatio']
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

    print_correlations('final-state', dataframe, 'EndCoverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("----------------------------------------------------------------------------------------------------")

    percent_reached = 1
    dataframe = ex.get_measurements(dataframe, percent_reached)
    print_correlations(str(percent_reached * 10) + 'p-time', dataframe, 'EndCoverage')

    logging.info("----------------------------------------------------------")

    percent_reached = 2
    dataframe = ex.get_measurements(dataframe, percent_reached)
    print_correlations(str(percent_reached * 10) + 'p-time', dataframe, 'EndCoverage')

    logging.info("----------------------------------------------------------------------------------------------------")
    logging.info("-- Measuring point at a certain percentage of the search budget")
    logging.info("-- All classes, that reached that point")
    logging.info("-- Remove zero branches")
    logging.info("----------------------------------------------------------------------------------------------------")

    percent_reached = 2
    dataframe = ex.get_measurements(dataframe, percent_reached)
    subset = dataframe[dataframe['Total_Branches'].gt(0)]
    print_correlations(str(percent_reached * 10) + 'p-time', subset, 'EndCoverage')


def foo_std(dataframe):
    groups = dataframe.groupby('TARGET_CLASS').agg({
        'EndCoverage': ['var', 'std', 'min', 'max', 'median'],
        'LOW_END_COVERAGE': 'mean',
        'Branchless': 'mean',
        'GradientRatio': 'mean',
        'BranchRatio': 'mean',
        'NotGradientRatio': 'mean',
        'InformationContent': 'mean',
        'NeutralityRatio': 'mean',
        'Fitness': 'mean'
    }).reset_index()

    groups[('EndCoverage', 'spread')] = groups[('EndCoverage', 'max')] - groups[('EndCoverage', 'min')]
    groups.sort_values(('EndCoverage', 'std'), inplace=True, ascending=False)

    fig, axs = plt.subplots(2, 3)
    draw_2d(axs[0, 0], fig, groups, 'GradientRatio at 20% (mean)', 'EndCoverage (std)', 'GradientRatio',
            ('EndCoverage', 'std'), title='a)')
    draw_2d(axs[0, 1], fig, groups, 'BranchRatio at 20% (mean)', 'EndCoverage (std)', 'BranchRatio',
            ('EndCoverage', 'std'), title='b)')
    draw_2d(axs[0, 2], fig, groups, 'NotGradRatio at 20% (mean)', 'EndCoverage (std)', 'NotGradientRatio',
            ('EndCoverage', 'std'), title='c)')
    draw_2d(axs[1, 0], fig, groups, 'IC at 20% (mean)', 'EndCoverage (std)', 'InformationContent',
            ('EndCoverage', 'std'), title='d)')
    draw_2d(axs[1, 1], fig, groups, 'NV/Gen at 20% (mean)', 'EndCoverage (std)', 'NeutralityRatio',
            ('EndCoverage', 'std'), title='e)')
    draw_2d(axs[1, 2], fig, groups, 'Fitness at 20% (mean)', 'EndCoverage (std)', 'Fitness', ('EndCoverage', 'std'),
            title='f)')
    plt.tight_layout()
    plt.show()


def foo_coverage(dataframe):
    fig, axs = plt.subplots(2, 3)
    draw_2d(axs[0, 0], fig, dataframe, 'GradientRatio at 20%', 'EndCoverage', 'GradientRatio', 'EndCoverage',
            title='a)')
    draw_2d(axs[0, 1], fig, dataframe, 'BranchRatio at 20%', 'EndCoverage', 'BranchRatio', 'EndCoverage', title='b)')
    draw_2d(axs[0, 2], fig, dataframe, 'NotGradRatio at 20%', 'EndCoverage', 'NotGradientRatio', 'EndCoverage',
            title='c)')
    draw_2d(axs[1, 0], fig, dataframe, 'IC at 20%', 'EndCoverage', 'InformationContent', 'EndCoverage', title='d)')
    draw_2d(axs[1, 1], fig, dataframe, 'NV/Gen at 20%', 'EndCoverage', 'NeutralityRatio', 'EndCoverage', title='e)')
    draw_2d(axs[1, 2], fig, dataframe, 'Fitness at 20%', 'EndCoverage', 'Fitness', 'EndCoverage', title='f)')
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

        ax.scatter(dataframe[x], dataframe[y], s=SCATTER_POINT_SIZE, c=dataframe[color].values, norm=norm,
                   cmap=colormap)
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
        print_correlation(name, dataframe, 'Fitness', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'BranchRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'GradientRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'NotGradientRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'InformationContent', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'NeutralityVolume', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'NeutralityRatio', referenceColumn, enablePlotting)
        print_correlation(name, dataframe, 'Generations', referenceColumn, enablePlotting)


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


def foo_analysis(dataframe):
    dataframe = ex.get_measurements(dataframe, 10)
    dataframe.info()

    features = ['Fitness', 'Total_Branches', 'GradientBranches', 'BranchRatio', 'NeutralityVolume', 'NeutralityRatio', 'InformationContent']

    for feature in features:
        ax = dataframe.hist(column=feature, bins=100)
        ax[0][0].set_ylabel("Count")
        ax[0][0].set_xlabel(feature)
        plt.title('Histogram - ' + feature + ' of $S_1$')
        plt.tight_layout()
        plt.show()

    corr =  dataframe[features].corr()
    fig, ax = plt.subplots()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=1, center=0, square=True, cbar_kws={"shrink": .5})

    plt.title('Correlation heatmap of $S_1$')
    plt.tight_layout()
    plt.show()


def setup_argparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(description="Collect the experiment results from the experiment_runner",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-results", help="The directory of the results", type=ex.check_dir_path, required=True)
    parser.add_argument("-only_pc", help="compares only classes with pc=yes", action='store_true')

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

    if args.only_pc:
        dataframe = dataframe[dataframe["_ParameterControlled"].eq("yes")]

    dataframe = ex.get_measurements(dataframe, -1)
    ex.print_result_infos(dataframe)

    logging.info("start evaluation...")

    # foo_percentage_dif(dataframe)
    # foo_general_infos(original, dataframe)
    foo_analysis(original)
    # foo_generations(dataframe)

    # logging.info("@20%")
    # dataframe = ex.get_measurements(dataframe, 2)

    # foo_correlation(dataframe)
    # foo_std(dataframe)
    # foo_coverage(dataframe)

    # ###################################################################

    # LOW_END_COVERAGE, HIGH_STDEV, Branchless, Coverage
    # _GradientRatio, _BranchRatio, _NotGradRatio
    # _InfoContent, _NeutralityGen
    # _Fitness
    # draw_3d(subset, 'BranchRatio at 20%', 'Fitness', 'EndCoverage', 'LOW_END_COVERAGE', 'BranchRatio', 'Fitness', 'EndCoverage', 'Well performing')
    # plt.show()


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
