'''
Desc
    Minimal set of code to produce a TC prediction matching the method used by
    BoM up to 2021/2022.
History
    2022 08 17 | Created by Jesse Greenslade

'''

import pandas as pd
import numpy as np

# LDA sklearn method
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

##################
### #CONSTANTS ###
##################


#Compute class scores
def add_class_column(df,target='AR'):
    median = df['AR'].median()
    df.loc[:,('class')] = df.apply(lambda row : classify_median(row, target, median),axis=1).values

def add_strings(a,b):
    """
    Use a vector function and pass in series of strings, see 
    https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6
    Made possible because newer pandas can return a series of datetime year/month/day strings
    """
    return a+b    


def classify_median(row,region,median):
    """
    Classify rows depending if they are above/below median
        class 1 if target <= median
        class 2 if target > median
    """
    rr = row[region]
    # need value, not series
    if isinstance(rr,pd.core.series.Series):
        rr = row[region][0] 
    if rr <= median:
        return 1
    else:
        return 2

def combo_method_SKL(X1,X2,Y,
                     prediction_year=None,
                     ):
    """
        ARGS:
            X1,X2: input pandas Series for LDA1 and LDA2 
                needs datetime index
                these are expected to be nino3.4 anom 3m mean and SOI 3m mean
            Y: pandas Series with TC counts and datetime index
            prediction_year=None : if not set assumed to be final year
            
        Returns:
            prediction_probabilities: numpy array[n_predictions,n_classes]
                row = predicted probabilities for each year in the 
                prediction range being below the median (class1)
                or above the median (class2)
    """
    ## Training and prediction ranges as list of indices
    if prediction_year is None:
        prediction_year = X1.index.year.values[-1]
    training_range = range(1970,prediction_year)
    i_tra = X1.index.year.isin(training_range)
    # final year of Y is being predicted, so array has 1 less length
    i_tra_Y = Y.index.year.isin(training_range)
    i_ver = X1.index.year.isin([prediction_year])
    # i_ver looks like [False,False,...,True] 
    # true where year == training range years
    
    ## X and y from data frame
    X_nino= X1.copy()[i_tra]
    X_nino_ver = X1.copy()[i_ver]
    X_soi = X2.copy()[i_tra]
    X_soi_ver = X2.copy()[i_ver]
    
    
    # y is based on median: 1 for less or equal, 2 for more than
    y0 = Y[i_tra_Y]
    median = y0.median()
    y = np.ones(len(y0),dtype=int)
    y[y0.values <= median] = 1
    y[y0.values > median] = 2
    
    ## Initialise LDA models
    LDA_soi = LinearDiscriminantAnalysis(solver='svd',priors=[.5,.5])
    LDA_nino = LinearDiscriminantAnalysis(solver='svd',priors=[.5,.5])
    
    ## fit models using training data
    LDA_soi.fit(X_soi.values.reshape(-1, 1), y)
    LDA_nino.fit(X_nino.values.reshape(-1, 1), y)
    
    ## Predict probability of class1,class2 for prediction input sets
    # reshape required as we are only using 1 input column
    prob_soi = LDA_soi.predict_proba(X_soi_ver.values.reshape(-1,1))
    prob_nino = LDA_nino.predict_proba(X_nino_ver.values.reshape(-1,1))

    # probabilities are given as array[n_predictions,n_classes]
    # so 2 columns, one for class 1, second for class 2
    #prob_combo = (prob_soi+prob_nino) / 2.0
    
    return prob_nino, prob_soi

def read_nino(
    url="http://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
    ):
    """
    Read NINO data from specified web URL - assumes NOAA CPC formatting
     - old url was "http://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii"
    """
    #Skip header - we will specify our own
    columns=['year','month','nino1+2','nino1+2 anom','nino3','nino3 anom','nino4','nino4 anom','nino3.4','nino3.4 anom']
    nino_data = pd.read_csv(url,skiprows=1,names=columns,delim_whitespace=True)
    #output nino3.4 anomalies in required format
    nino_data['date'] =  pd.to_datetime(add_strings(nino_data['year'].map(str),nino_data['month'].map(str)),format='%Y%m')
    nino_data.set_index('date',inplace=True)
    # remove superfluous year, month columns
    nino_data.drop(['year','month'],axis='columns',inplace=True)
    nino_data.sort_index(inplace=True)

    for column in [
        'nino1+2', 'nino1+2 anom',
        'nino3', 'nino3 anom',
        'nino4', 'nino4 anom',
        'nino3.4', 'nino3.4 anom',
        ]:
        # take 3m rolling mean as new column (column index = 2, doesn't matter)
        nino_data.insert(2,column+' 3m mean', nino_data[column].rolling(window=3).mean(), allow_duplicates=True)

    return nino_data

def read_oni(
    url="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
):
    """
    Read NOAA ONI data.

    Returns a dataframe indexed by month containing:
        oni
        nino3.4
    """

    nino_data = pd.read_csv(
        url,
        delim_whitespace=True
    )

    season_to_month = {
        'DJF': 2,
        'JFM': 3,
        'FMA': 4,
        'MAM': 5,
        'AMJ': 6,
        'MJJ': 7,
        'JJA': 8,
        'JAS': 9,
        'ASO': 10,
        'SON': 11,
        'OND': 12,
        'NDJ': 1
    }

    nino_data['month'] = nino_data['SEAS'].map(season_to_month)

    nino_data['date'] = pd.to_datetime(
        {
            'year': nino_data['YR'],
            'month': nino_data['month'],
            'day': 1
        }
    )

    nino_data.rename(
        columns={
            'ANOM': 'anom',
            'TOTAL': 'total'
        },
        inplace=True
    )

    nino_data.set_index('date', inplace=True)

    nino_data = nino_data[['total', 'anom']]

    nino_data.sort_index(inplace=True)

    return nino_data

def read_roni(
    url="https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt",
):
    """
    Read NOAA RONI data.

    Returns DataFrame indexed by month with:
        roni
    """

    roni_data = pd.read_csv(
        url,
        delim_whitespace=True
    )

    season_to_month = {
        'DJF': 2,
        'JFM': 3,
        'FMA': 4,
        'MAM': 5,
        'AMJ': 6,
        'MJJ': 7,
        'JJA': 8,
        'JAS': 9,
        'ASO': 10,
        'SON': 11,
        'OND': 12,
        'NDJ': 1,
    }

    roni_data['month'] = roni_data['SEAS'].map(season_to_month)

    roni_data['date'] = pd.to_datetime(
        {
            'year': roni_data['YR'],
            'month': roni_data['month'],
            'day': 1,
        }
    )

    roni_data.rename(
        columns={'ANOM': 'anom'},
        inplace=True
    )

    roni_data.set_index('date', inplace=True)

    roni_data = roni_data[['anom']]

    roni_data.sort_index(inplace=True)

    return roni_data

def read_processed_tc(TC_folder='data/2021-22/'):
    """
    Read data used by 2021/2022 seasonal prediction (files created to be read by fortran)
    """
    data=None
    tc_filemap = { # fname : region
        "OZ90_160.data":"AR",
        "OZ90_125.data":"AR-W",
        "OZ105_130_25S.data":"AR-NW",
        "OZ125_1425.data":"AR-N",
        "OZ1425_160.data":"AR-E",
        "SPO1425_240.data":"SPO",
        "SPO1425_165.data":"SPO-W",
        "SPO165_240.data":"SPO-E",
        }
    for fname,region in tc_filemap.items():
        f = TC_folder+fname
        columns=['dstr',region]
        fdata = pd.read_csv(f,skiprows=0,names=columns,delim_whitespace=True, dtype={'dstr':str,region:int})
        # yyyymmddyyyymmdd TCcount, just want yyyymm for september each year
        fdata['date0'] =  fdata['dstr'].str[:6]
        fdata['date'] =  pd.to_datetime(fdata['date0'],format='%Y%m')
        fdata.set_index('date',inplace=True)
        # remove superfluous year, month columns
        fdata.drop(['dstr',"date0"],axis='columns',inplace=True)
        
        # merge for return dataset
        if data is None:
            data = fdata
        else:
            # merge with data
            data = pd.merge(data,fdata,left_index=True,right_index=True,how='outer')

    # Was originally just using September   
    #data = data[data.index.month == 9].copy()
        
    return data

def read_soi(url="http://www.bom.gov.au/clim_data/IDCKGSH000/soi_monthly.txt"):
    """
    default is BOM maintained document: http://www.bom.gov.au/clim_data/IDCKGSH000/soi_monthly.txt
    Return a Dataframe based on the input text file with datetime index
    """
    df = pd.read_csv(url,header=None,names=['date','SOI'])
    #Declare empty dataframe
    df['datetime']=pd.to_datetime(df.date,format='%Y%m')
    df.set_index('datetime',inplace=True)
    
    for column in ['SOI']:
        df.insert(2,column+' 3m mean', df[column].rolling(window=3).mean(), allow_duplicates=True)

    return df
