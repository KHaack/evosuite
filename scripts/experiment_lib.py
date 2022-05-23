import logging
import os
import pandas as pd

# files and directories
FILE_CLASSES = "samples\\00 - original - 23894.txt"
FILE_EXPORT = "samples\\export.txt"
PATH_WORKING_DIRECTORY = "C:\\Users\\kha\\Desktop\\Benchmark"
# filtering
FILTER_ZERO_GENERATIONS = True
FILTER_PERCENTAGE = True


def createExport(dataframe):
    """
    Export the passed dataframe to the runner format.
    :param dataframe: The dataframe to export
    :return: None
    """
    pathSamples = os.path.join(PATH_WORKING_DIRECTORY, FILE_CLASSES)
    pathExport = os.path.join(PATH_WORKING_DIRECTORY, FILE_EXPORT)

    samples = pd.read_csv(pathSamples, delimiter='\t', names=['project', 'TARGET_CLASS'])
    merged = dataframe.merge(samples, left_on='TARGET_CLASS', right_on='TARGET_CLASS', how='inner')
    merged = merged.groupby(['project', 'TARGET_CLASS']).count().reset_index()[['project', 'TARGET_CLASS']]

    merged.to_csv(pathExport, header=None, sep='\t', index=False)


def addAdditionalColumns(dataframe):
    dataframe['PercentageReached'] = dataframe['NeutralityVolume'].str.count(';') * 10

    # classes
    classes = {'worst': 0, 'bad': 1, 'mid': 2, 'good': 3, 'very good': 4}
    dataframe.loc[dataframe['Coverage'].le(0.20), 'class'] = classes['worst']
    dataframe.loc[dataframe['Coverage'].gt(0.20) & dataframe['Coverage'].le(0.40), 'class'] = classes['bad']
    dataframe.loc[dataframe['Coverage'].gt(0.40) & dataframe['Coverage'].lt(0.60), 'class'] = classes['mid']
    dataframe.loc[dataframe['Coverage'].ge(0.60) & dataframe['Coverage'].lt(0.80), 'class'] = classes['good']
    dataframe.loc[dataframe['Coverage'].ge(0.80), 'class'] = classes['very good']

    dataframe.loc[dataframe['class'].lt(4), 'groundTruth'] = True
    dataframe.loc[dataframe['class'].eq(4), 'groundTruth'] = False

    return dataframe


def clean(dataframe):
    # remove
    dataframe.drop('_FitnessMax', axis=1, inplace=True)
    dataframe.drop('_FitnessMin', axis=1, inplace=True)
    dataframe.drop('Fitness', axis=1, inplace=True)
    dataframe.drop('criterion', axis=1, inplace=True)
    dataframe.drop('Total_Goals', axis=1, inplace=True)
    dataframe.drop('Covered_Goals', axis=1, inplace=True)
    dataframe.drop('BranchCoverage', axis=1, inplace=True)
    dataframe.drop('Generations', axis=1, inplace=True)

    # rename
    dataframe.rename({
        '_InformationContent': 'InformationContent',
        '_NeutralityVolume': 'NeutralityVolume',
        '_FitnessRatio': 'Fitness',
        '_Generations': 'Generations',
        '_GradientBranchesCovered': 'GradientBranchesCovered',
        '_GradientBranches': 'GradientBranches'
    }, inplace=True, axis=1)

    return dataframe


def getNth(x, n):
    """
    Returns the n-th element of the passed string x in the format [a;b;c;d].
    :param x: String in the format [a;b;c;d]
    :param n: The n that should be returned.
    :return: The n-th element.
    """
    if x == '[]':
        return

    x = x.replace('[', '').replace(']', '')
    parts = x.split(';')

    if len(parts) <= n:
        return

    return float(parts[n])

def getMeasurings(dataframe, percent):
    """
    Extract the measuring.
    :param dataframe: The dataframe.
    :param percent:
    :return:
    """
    # filter
    if percent >= 0:
        dataframe = dataframe[dataframe['PercentageReached'].ge(percent * 10)]

    index = percent - 1

    # get measurings
    dataframe['_GradientBra'] = dataframe['GradientBranches'].apply(lambda x: getNth(x, index))
    dataframe['_GradientCov'] = dataframe['GradientBranchesCovered'].apply(lambda x: getNth(x, index))
    dataframe['_Fitness'] = dataframe['Fitness'].apply(lambda x: getNth(x, index))
    dataframe['_InfoContent'] = dataframe['InformationContent'].apply(lambda x: getNth(x, index))
    dataframe['_NeutralityVol'] = dataframe['NeutralityVolume'].apply(lambda x: getNth(x, index))
    dataframe['_Generations'] = dataframe['Generations'].apply(lambda x: getNth(x, index))

    # calculate others
    dataframe.loc[dataframe['_Generations'].eq(0), '_NeutralityGen'] = 0
    dataframe.loc[dataframe['_Generations'].gt(0), '_NeutralityGen'] = dataframe['_NeutralityVol'] / (
        dataframe['_Generations'])
    dataframe.loc[dataframe['_NeutralityGen'] > 1, '_NeutralityGen'] = 1

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_NotGradRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_NotGradRatio'] = (dataframe['Total_Branches'] - dataframe[
        '_GradientBra']) / (dataframe['Total_Branches'])

    dataframe.loc[dataframe['_GradientBra'].eq(0), '_GradientRatio'] = 0
    dataframe.loc[dataframe['_GradientBra'].gt(0), '_GradientRatio'] = dataframe['_GradientCov'] / (
        dataframe['_GradientBra'])

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_BranchRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_BranchRatio'] = dataframe['_GradientBra'] / (
        dataframe['Total_Branches'])

    dataframe.loc[dataframe['Total_Branches'].eq(0), '_NotCovGraRatio'] = 0
    dataframe.loc[dataframe['Total_Branches'].gt(0), '_NotCovGraRatio'] = (dataframe['_GradientBra'] - dataframe['_GradientCov']) / (
        dataframe['Total_Branches'])

    return dataframe


def filter(dataframe, minExecutions):
    """
    Filter the passed dataframe.
    :param dataframe: The dataframe to filter.
    :return: The filtered dataframe.
    """
    # not 10% reached
    if FILTER_PERCENTAGE:
        totalLength = len(dataframe.index)
        dataframe = dataframe[dataframe['PercentageReached'] > 0]
        logging.info("Tests not reached 10%:\t" + str(totalLength - len(dataframe.index)))

    # zero generations
    if FILTER_ZERO_GENERATIONS:
        totalLength = len(dataframe.index)
        dataframe = dataframe[dataframe['_Generations'] > 0]
        logging.info("Zero generations tests:\t" + str(totalLength - len(dataframe.index)))

    # executions
    if minExecutions > 0:
        totalLength = len(dataframe.index)
        groups = dataframe.groupby('TARGET_CLASS').count()
        groups = groups.reset_index()
        groups = groups[groups['Algorithm'] >= minExecutions]
        dataframe = dataframe[dataframe['TARGET_CLASS'].isin(groups['TARGET_CLASS'])]
        logging.info(f"Tests less then {str(minExecutions)}execs:\t{str(totalLength - len(dataframe.index))}")

    return dataframe