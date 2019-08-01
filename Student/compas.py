""" 
Author: Thomas Cintra
Class: CS 181R
Week 9 - Algorithmic Bias
Homework 5
Name:
"""

# Import all significant modules. cleanCSV is a file you  are provided with to generate Compas.csv
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

    ### ========== TODO : QUESTION 1 ========== ###

    # Call pie(df, column)


    ### ========== TODO : END ========== ###

    ### ========== TODO : QUESTION 2 ========== ###

    # Plot threat score distplots


    ### ========== TODO : END ========== ###

    print()

    ### ========== TODO : QUESTION 3 ========== ##

    # Call confMatrix(df, low, high, column, var)


    ### ========== TODO : END ========== ###

    print()

    ### ========== TODO : QUESTION 4 ========== ##

    # Call class_imbalance(df, column, var)


    ### ========== TODO : END ========== ###

    print()

    ### ========== TODO : QUESTION 5 ========== ##

    # Call opThresh(metric, column, var)


    ### ========== TODO : END ========== ###

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


    # Assign your pie chart to ax. Remember we are using matplotlib's pie function. 


    # Title your plot using plt.title.


    ### ========== TODO : END ========== ###
    plt.show()

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
    Nothing
    """
    ### ========== TODO : QUESTION 2 ========== ###

    # Sets the dimensions for your plot.
    plt.figure(figsize = (18, 9))

    # Create subsets of data containing only individuals with var1 or var2.

    # Index your subsets further to generate your 6 plots.


    # Using plt.subplot, create your seaborn distplot. You might want to label the y axis on SOME of them


    ### ========== TODO : END ========== ###
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
    ### ========== TODO : QUESTION 3 ========== ###
    
    # Use theat to create a new df with the new 'Assessment' column.

    if column:
        # If column is provided, index your df. Delete the pass statement.


        pass

    # Assign a y_true and y_pred and generate your confusion matrix. Use it to find TP, TN, FP, FN


    if metric:
        return FP/(TN + FP), FN/(TP + FN), f1_score(testData, predData), roc_auc_score(testData, predData)

    ### ========== TODO : END ========== ###

    # Prints all your information
    print('Confusion Matrix For', var,':')
    print(confusion_Matrix) 
    print('Accuracy Score :',accuracy_score(testData, predData))
    print('Labelled High Risk But Didn\'t Reoffend: ', FP/(TN + FP))
    print('Labelled Low Risk But Did Reoffend: ', FN/(TP + FN))
    print("Number of", var, "people surveyed:", len(pop))
    return

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
    ### ========== TODO : QUESTION 4 ========== ###

    if column:
        # If column argument is provided, find class imbalance by indexing. Delete pass statement.


        pass

    else: 
        # If column argunent is not provided, find class imabalance for the whole dataset. Delete pass statement.
        
        
        pass

    ### ========== TODO : END ========== ###
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
    ### ========== TODO : QUESTION 5 ========== ###

    # Find a way to store information as you loop through combinations of low/high boundaries. There are other ways to approach this problem.


    # Some metrics require maximization, some minimization. Determine which is which here. Delete pass statement.
    if metric == metric1_FPR or metric == metric2_FNR:


        pass

    elif metric == metric3_F1 or metric == metric4_auc_roc:


        pass
    
    # Titles and prints your information. MUST edit print statement to include optimal boundary.
    if var == None:
        var = 'all'
    print("The optimal threshhold for", var, "inmates using", metric, "is:", min_max())

    ### ========== TODO : END ========== ###

# --------------------------------- END --------------------------------- #

if __name__ == "__main__":
    main()