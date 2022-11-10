import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pre

import env

###################################################################################
#################################### ACQUIRE DATA #################################
###################################################################################

def get_db_url(db, user=env.username, password=env.password, host=env.host):
    '''
    This function uses the imported host, username, password from env file, 
    and takes in a database name and returns the url to access that database.
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}' 

def new_zillow_data():
    '''
    This reads the zillow 2017 properties data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = 261;
                '''
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url(db = 'zillow'))

    return df

def aquire_zillow_data(new = False):
    ''' 
    Checks to see if there is a local copy of the data, 
    if not or if new = True then go get data from Codeup database
    '''
    
    filename = 'zillow.csv'
    
    #if we don't have cached data or we want to get new data go get it from server
    if (os.path.isfile(filename) == False) or (new == True):
        df = new_zillow_data()
        #save as csv
        df.to_csv(filename,index=False)

    #else used cached data
    else:
        df = pd.read_csv(filename)
          
    return df


###################################################################################
##################################### CLEAN DATA ##################################
###################################################################################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    for col in col_list:

        q1, q3 = df[col].quantile([0.25, 0.75]) # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def clean_zillow(df, remove = True):
    '''Takes in zillow data and returns a clean df'''

    #make column names more human readable
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'square_feet',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built'})
    
    #if remove == True drop outliers
    if remove == True:
        df = remove_outliers(df, 1.5, ['bedrooms','bathrooms','square_feet', 
                                   'tax_value', 'taxamount'])  
        
    #drop nulls
    df = df.dropna()
    
    return df

def wrangle_zillow(new = False, remove = True):
    ''' 
    Checks to see if there is a local copy of the data, 
    if not or if new = True then go get data from Codeup database
    Then prepares the data by making feature names human readable
    if remove = True it removes outliers and finally drops the leftover nulls.
    '''
    
    if new == True:
        df = aquire_zillow_data(new == True)
    else:
        df = aquire_zillow_data()
    
    if remove == False:
        df = clean_zillow(df, remove = False)
    else:
        df = clean_zillow(df)
    
    return df

def scale_zillow(train, validate, test, scale_features=['bedrooms', 'bathrooms', 'square_feet', 'taxamount']):
    '''
    Takes in train, validate, test and a list of features to scale
    and scales those features.
    Returns df with new columns with scaled data
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    quantile = pre.QuantileTransformer(output_distribution='normal')
    quantile.fit(train[scale_features])
    
    train_scaled[scale_features] = pd.DataFrame(quantile.transform(train[scale_features]),
                                                  columns=train[scale_features].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[scale_features] = pd.DataFrame(quantile.transform(validate[scale_features]),
                                                  columns=validate[scale_features].columns.values).set_index([validate.index.values])
    
    test_scaled[scale_features] = pd.DataFrame(quantile.transform(test[scale_features]),
                                                 columns=test[scale_features].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

###############################################
################# SPLIT DATA ##################
###############################################

def split_data_other(df, train_size=0.65, validate_size=0.2):
    '''
    Takes in a data frame, the train size and validate size
    It returns train, validate, and test data frames based on the sizes
    that were passed.
    '''
    train, test = train_test_split(df, train_size = train_size + validate_size , random_state=27)
    train, validate = train_test_split(train, test_size = validate_size/(train_size + validate_size), random_state=27)
    
    return train, validate, test

def split_data(df, test_size=0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(df, test_size = test_size , random_state=27)
    train, validate = train_test_split(train, test_size = (test_size + 0.05)/(1-test_size), random_state=27)
    
    return train, validate, test

def train_validate_test_split(df):
    '''
    Takes in a data frame and the target variable column  and returns
    train (65%), validate (20%), and test (15%) data frames.
    '''
    train, test = train_test_split(df,test_size = 0.15, random_state=27)
    train, validate = train_test_split(train, test_size = 0.235, random_state=27)
    
    return train, validate, test