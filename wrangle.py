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