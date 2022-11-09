import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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