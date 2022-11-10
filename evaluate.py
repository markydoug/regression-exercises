import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metric
from math import sqrt


def plot_residuals(y, yhat):
    '''
    Takes in  y and yhat and creates a residual plot
    '''
    residuals = y - yhat
    sns.scatterplot(x=y, y=residuals)
    plt.title('Residuals vs. y')
    plt.show()

def regression_errors(y, yhat):
    '''
    Takes in y and yhat and returns sum of squared errors (SSE) 
    explained sum of squares (ESS), total sum of squares (TSS) 
    mean squared error (MSE), and root mean squared error (RMSE)
    '''

    MSE = metric.mean_squared_error(y, yhat)
    SSE = MSE*len(y)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    RMSE = sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    '''
    Takes in y and returns SSE, MSE and RMSE for baseline
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    MSE_baseline = metric.mean_squared_error(y, baseline)
    SSE_baseline = MSE_baseline*len(y)
    RMSE_baseline = sqrt(MSE_baseline)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    '''
    Takes in y and yhat and compares SSE and SSE_baseline and 
    returns True if your model is better than baseline 
    and False if it is not.
    '''
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        return print(f'SSE: {SSE}\nSSE_baseline: {SSE_baseline}\nThis model performs better than baseline.')
    else:
        return print(f'SSE: {SSE}\nSSE_baseline: {SSE_baseline}\nThis model performs worse than baseline.')
    