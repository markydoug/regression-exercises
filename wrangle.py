import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    This function reads the telco data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                JOIN propertylandusetype USING (propertylandusetypeid)
                WHERE propertylandusedesc LIKE "Single Family Residential";
                '''
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url(db = 'zillow'))

    return df

def aquire_zillow_data(new = False):
    ''' Aquires data for project'''
    
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