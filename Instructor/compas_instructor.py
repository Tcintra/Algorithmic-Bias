""" 
Author: Thomas Cintra
Class: CS 181R
Week 9 - Algorithmic Bias
Homework 5
Name:
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cleanCSV
from importlib import reload
reload(cleanCSV)

path = os.path.join("..", "Data")

# --------------------------------- START --------------------------------- #

def main():

    print()

    # Run cleanCSV.py to create a cleaned Compas.csv in the Data folder
    cleanCSV.clean()

    # Read Compas.csv onto df
    df = pd.read_csv(os.path.join(path, 'Compas.csv'))

    # Call pie(df, column)
    pie(df, 'Race')

    # Plot threat score comparison
    threat_scores_dists(df, 'Race', 'African-American', 'Caucasian')

    print()

    # Call confMatrix
    confMatrix(df, 4, 5, 'Gender', 'Female')

    print()

    class_imbalance(df, 'Race', 'Caucasian')

    print()

    # Call opThresh
    opThresh(metric1_FPR, column = 'Gender', var = 'Female')
    opThresh(metric2_FNR, column = 'Gender', var = 'Female')
    opThresh(metric3_F1, column = 'Gender', var = 'Female')
    opThresh(metric4_auc_roc, column = 'Gender', var = 'Female')

def pie(df, column):
    """
    pie(...) generates a pie chart for the distribution of unique values of column in df.

    Parameters
    ----------
    df : DataFrame
    column : column in df

    Returns
    -------
    Nothing
    """
    ### ========== TODO : QUESTION 1 ========== ###

    # Sets the dimensions for your plot.
    plt.figure(figsize=(5,5))

    # Create an array of data and assign it to a variable such as x, which will serve as the data for your pie chart.
    x = df[column].value_counts() / len(df)

    # Assign your pie chart to ax. Remember we are using matplotlib's pie function. 
    ax  = plt.pie(x, labels = list(x.keys()))

    # Title your plot using plt.title.
    plt.title('Pie Chart for ' + column)

    ### ========== TODO : END ========== ###
    plt.show()

def class_imbalance(df, column = None, var = None):
    """
    class_imbalance(...) prints a Pandas series detailing the proportion of individuals who reoffended and who didn't.
    If column and var are provided, class_imbalance(...) only considers individuals with the feature var in column.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    Nothing
    """
    if column:
        print('Imbalance in Testing Set for ' + var + 's:')
        pop = df.loc[df[column] == var]
        print(pop['Recidivism'].value_counts() / len(pop))

    else: 
        print('Imbalance in Testing Set:')
        print(df['Recidivism'].value_counts() / len(df))

    return
    

def threat_scores_dists(df, column, var1, var2):
    """
    threat_scores_dists(...) generates 6 plots for the distribution of COMPAS threat scores.
    The first three graphs belong to individuals with the feature var1.
    The second three graphs belong to individuals with the feature var2.
    Each of these three corresponds to (1: all individuals, 2: those who reoffended, 3: those who didn't reoffend)


    Parameters
    ----------
    df : DataFrame
    column : column in df
    var1 : unique value in column
    var2 : another unique value in column

    Returns
    -------
    Six distplots as detailed above
    """
    plt.figure(figsize = (18, 9))

    pop1 = df.loc[df[column] == var1]
    pop2 = df.loc[df[column] == var2]

    group = pop1, pop1.loc[pop1['Recidivism'] == 1], pop1.loc[pop1['Recidivism'] == 0], pop2, pop2.loc[pop2['Recidivism'] == 1], pop2.loc[pop2['Recidivism'] == 0]
    title = var1, var1 + 's who Reoffended', var1 + 's who didn\'t Reoffend' , var2, var2 + 's who Reoffended', var2 + 's who didn\'t Reoffend'

    for i in range(len(group)):
        plt.subplot(2, 3, i+1)
        ax = sns.distplot(group[i]['Threat Score'], bins = 10, norm_hist = True, kde = False).set_title(title[i])
        plt.xlim(1,10)
        if i == 0 or i == 3:
            plt.ylabel('Normed Frequency')
    plt.show()

def threat(df, low, high, opThresh = None):
    """
    threat(...) is a helper function that edits the column 'Assessment' in df.
    A clean copy of Compas.csv has unique values: 'Low', 'Medium', 'High'.
    threat(...) uses the low and high parameters to assign 0s to low risk offenders and 1s to high risk offenders.
    threat(...) then returns df with the new 'Assessment' column.
    (OPTIONAL) The opThresh parameter is used by opThresh to find the optimal low/high boundaries based on a given metric.

    Parameters
    ----------
    df : DataFrame
    low : Boundary for 0s assingments
    high : Boundary for 1s assignments
    opThresh : (OPTIONAL) turns threat(...) into a helper function for opThresh(...)

    Returns
    -------
    df with new 'Assessment' column 
    ndf if opThresh
    """
    def threshold(row):
        if row['Threat Score'] <= low:
            Threat = 0
        elif row['Threat Score'] >= high:
            Threat = 1
        else:
            Threat = np.NaN
        return Threat
    df['Assessment'] = df.apply(threshold, axis = 1)
    if opThresh:
        ndf = df.dropna()
        return ndf
    else: 
        df.dropna(inplace = True)
        return df

def confMatrix(df,low, high, column = None, var = None, metric = None):
    """
    confMatrix(...) uses threat(...) to print a confusion matrix for df.
    confMatrix(...) also prints Accuracy score, False Positive Rates, False Negative Rates, and total individuals surveyed.
    The columns 'Recidivism' and 'Assessment' as y_true and y_pred respectively.
    (OPTIONAL) column and var arguments make a confusion matrix exclusively for individuals with the feature var in column.
    (OPTIONAL) metric argument turns confMatrix into a helper function for opThresh(...)

    Parameters
    ----------
    df : DataFrame
    low : Boundary for 0s assignments
    high : Boundary for 1s assignments
    (OPTIONAL) column : column in DataFrame
    (OPTIONAL) var : unique value in column
    (OPTIONAL) metric : metric changes the return statement of confMatrix(...) for opThresh(...)

    Returns
    -------
    if metric == True: returns FPR, FNR, F1 score, AUC ROC score
    else:  nothing
    """
    ndf = threat(df, low, high)
    pop = ndf
    if column:
        pop = ndf.loc[ndf[str(column)] == var]
    testData = pop['Recidivism']
    predData = pop['Assessment']
    confusion_Matrix = confusion_matrix(testData, predData)
    TP = confusion_Matrix[1][1]
    TN = confusion_Matrix[0][0]
    FP = confusion_Matrix[0][1]
    FN = confusion_Matrix[1][0]
    if metric:
        return FP/(TN + FP), FN/(TP + FN), f1_score(testData, predData), roc_auc_score(testData, predData)
    print('Confusion Matrix For', var,':')
    print(confusion_Matrix) 
    print('Accuracy Score :',accuracy_score(testData, predData))
    print('Labelled High Risk But Didn\'t Reoffend: ', FP/(TN + FP))
    print('Labelled Low Risk But Did Reoffend: ', FN/(TP + FN))
    print("Number of", var, "people surveyed:", len(pop))
    return

def metric1_FPR(low, high, column, var):
    """
    metric1_FPR(...) is a helper function for opThresh.
    metric1_FPR(...) returns the False Positive Rates from confMatrix(...) with specified low/high arguments.
    opThresh(...) then minimizes FPR to find the optimal threshold for df.

    Parameters
    ----------
    low : Boundary for 0s assignments
    high : Boundary for 1s assignments
    column : column in df
    var : unique value in column

    Returns
    -------
    False Positive Rate
    """
    df = pd.read_csv(os.path.join(path, 'Compas.csv'))
    FPR = confMatrix(df,low, high, column, var, metric = True)[0]
    return FPR
    
def metric2_FNR(low, high, column, var):
    """
    metric1_FNR(...) is a helper function for opThresh.
    metric1_FNR(...) returns the False Negative Rates from confMatrix(...) with specified low/high arguments.
    opThresh(...) then minimizes FNR to find the optimal threshold for df.

    Parameters
    ----------
    low : Boundary for 0s assignments
    high : Boundary for 1s assignments
    column : column in df
    var : unique value in column

    Returns
    -------
    False Negative Rate
    """
    df = pd.read_csv(os.path.join(path, 'Compas.csv'))
    FNR = confMatrix(df, low, high, column, var, metric = True)[1]
    return FNR
 
def metric3_F1(low, high, column, var):
    """
    metric1_F1(...) is a helper function for opThresh.
    metric1_F1(...) returns the F1 score from confMatrix(...) with specified low/high arguments.
    opThresh(...) then maximizes F1 score to find the optimal threshold for df.

    Parameters
    ----------
    low : Boundary for 0s assignments
    high : Boundary for 1s assignments
    column : column in df
    var : unique value in column

    Returns
    -------
    F1 score
    """
    df = pd.read_csv(os.path.join(path, 'Compas.csv'))
    F1 = confMatrix(df,low, high, column, var, metric = True)[2]
    return F1

def metric4_auc_roc(low, high, column, var):
    """
    metric1_auc_roc(...) is a helper function for opThresh.
    metric1_auc_roc(...) returns the AUC ROC score from confMatrix(...) with specified low/high arguments.
    opThresh(...) then minimizes AUC ROC to find the optimal threshold for df.

    Parameters
    ----------
    low : Boundary for 0s assignments
    high : Boundary for 1s assignments
    column : column in df
    var : unique value in column

    Returns
    -------
    AUC ROC score
    """
    df = pd.read_csv(os.path.join(path, 'Compas.csv'))
    score = confMatrix(df,low, high, column, var, metric = True)[3]
    return score

def opThresh(metric, column = None, var = None):
    """
    opThresh(...) prints the optimal low/high boundaries for df with respect to a  given metric.

    Parameters
    ----------
    metric : details which metric opThresh(...) should optimize for
    (OPTIONAL) column : column in df
    (OPTIONAL) var : unique value in column

    Returns
    -------
    Nothing
    """
    dict = {}
    for low in range(1, 10):
        for high in range(low + 1, 11):
            dict[low, high] = metric(low, high, column, var)
    if metric == metric1_FPR or metric == metric2_FNR:
        min_max = min
    elif metric == metric3_F1 or metric == metric4_auc_roc:
        min_max = max
    if var == None:
        var = 'all'
    print("The optimal threshhold for", var, "inmates using", metric, "is:", min_max(dict, key = dict.get))

# --------------------------------- END --------------------------------- #

if __name__ == "__main__":
    main()