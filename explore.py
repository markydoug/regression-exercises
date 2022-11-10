import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


###################################################################################
################################## EXPLORE DATA ###################################
###################################################################################

def explore_uvar(df, cont_vars, cat_vars):
    explore_cont_uvar(df, cont_vars)
    explore_cat_uvar(df, cat_vars)


def explore_cont_uvar(df, cont_vars):
    '''
    Takes in a data frame and a list of continuous variables
    Returns univarite stats for numerical data
    '''
    for col in cont_vars:
        print(col)
        print(df[col].describe())
        df[col].hist()
        plt.show()
        sns.boxplot(y=col, data=df)
        plt.show()

def explore_cat_uvar(df, cat_vars):
    '''
    Takes in a data frame and a list of categorical variables
    Returns univarite stats
    '''
    for col in cat_vars:
        print(col)
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True)*100)
        sns.countplot(x=col, data=df)
        plt.show()

def explore_bvar(df, target, cont_vars, cat_vars):
    explore_cat_bvar(df, target, cont_vars)
    explore_cont_bvar(df, target, cat_vars)

def explore_cont_bvar(df, target, cont_vars):
    '''
    Takes in a data frame, target variable, and a 
    list of continuos variables. Returns bivarite stats 
    '''
    for col in cont_vars:
        sns.barplot(x=target, y=col, data=df)
        rate = df[col].mean()
        plt.axhline(rate,  label = f'Overall Mean of {col}', linestyle='dotted', color='black')
        plt.legend()
        plt.show()

def explore_cat_bvar(df, target, cat_vars):
    '''
    Takes in a data frame, target variable, and a 
    list of categorical variables. Returns bivarite stats 
    '''
    for col in cat_vars:
        sns.barplot(x=col, y=target, data=df)
        rate = df[target].mean()
        plt.axhline(rate, label = f'Average {target} rate', linestyle='--', color='black')
        plt.legend()
        plt.show()

def plot_variable_pairs(df, target):
    '''
    Takes in a dataframe and target variable and plots each feature 
    with the target variable
    '''
    cols = df.columns.to_list()
    cols.remove(target)
    for col in cols:
        sns.lmplot(x=col, y=target, data=df, line_kws={'color': 'red'})
        plt.show()

def plot_categorical_and_continuous_vars(df, cat_vars, cont_vars):
    '''
    Takes in a dataframe and a list of the columns that hold 
    the continuous and categorical features and outputs a boxplot, 
    a violinplot, and a barplot comparing continuous and categorical features.
    '''
    for col in cat_vars:
        for col2 in cont_vars:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,6))
            fig.suptitle(f'{col} vs. {col2}')
            sns.boxplot(data=df, x=col, y=col2, ax=ax1)
            sns.violinplot(data=df, x=col, y=col2, ax=ax2)
            sns.barplot(data=df, x=col, y=col2, ax=ax3)
            plt.show()