
'''
Desc
    Code used to analyse prediction.py TC Predictions
    Also can add to and break this file without breaking prediction.py
History
    17/08/2022 Made by Jesse Greenslade
'''
import pandas as pd
import numpy as np
import os, re
import datetime as dt

from sklearn.model_selection import LeaveOneOut

#from prediction import *
import prediction


## CONSTANTS

## METHODS

def brier_skill_score(obs,prob_below, 
    assumed_BS_ref_prob_below=.5, show_calculations=False,
    ):
    """ 
        Brier Skill is essentially mean square error:
            (prediction - [0 if true, 1 if false])**2
        BSS = 1 - BS/BS_ref
        BS_ref = score of reference method we are trying to beat
            in this case BS_ref is an assumed 50% either way
        
        BSS = 1 the forecast has perfect skill compared to climatology;
        BSS = 0 the forecast has no skill compared to climatology);
        BSS = a negative value the forecast is less accurate than climatology.
        
        ARGUMENTS:
            prob_below is odds of class 1
            obs is list of class 1 or class 2 occurrences
            assumed_... is reference climatological model estimate(s) of class 1
                this could be updated to (for example) the number of class 1 outcomes / total number of predictions
                currently is set to 50% (expect half to be greater than median, half to be = or below)
    """
    # if prob_below is [n rows, 2 columns] then it is probably [prob_below;prob_above]
    # fix it with warning
    if len(np.shape(prob_below)) == 2 and np.shape(prob_below)[1]==2:
        print("WARNING: brier_skill_score prob_below is assuming the first column to be probability of class 1")
        prob_below = prob_below[:,0]

    # remove any missing values
    df = pd.DataFrame({'obs':obs,'prob_below':prob_below})
    df = df.dropna()
    # Skill: 0 if 100% prediction is correct, 0 if 100% prediction is wrong
    df['BS'] = (df['prob_below'] - (df['obs']==1).astype(int))**2
    df['BS_ref'] = (assumed_BS_ref_prob_below - (df['obs']==1).astype(int))**2
    df['BSS'] = 1 - df['BS']/df['BS_ref']
    if show_calculations:
        print(df)
    # return mean of BSS
    BSS = df['BSS'].mean()
    ## TEST/CHECK with following
    # print("Great model score:")
    # print(brier_skill_score([1,1,2,2],[.9,.9,.1,.1],show_calculations=True)) # all very correct

    # print("horrid model score:")
    # print(brier_skill_score([1,1,2,2],[.3,.4,.8,.7],show_calculations=True)) # all quite incorrect

    # print("midrange model score:")
    # print(brier_skill_score([1,1,2,2],[.6,.7,.6,.2],show_calculations=True)) # right, right, wrong, right

    # print("prob_below is two columns")
    # print(brier_skill_score([1,1,2,2],np.array([[.6,.7,.6,.2],[.4,.3,.4,.8]]).T,show_calculations=True)) # right, right, wrong, right
    return BSS

def calculate_classed_leps_skill(y_obs, probabilities):
    """
    SCORE based on Linear Error in Probability Space
        1/n * sum (p_i(class1) - p_i(class2))
        p_i is probability for sample i, class 1 is less than median, 
        class 2 is greater or equal to median
    Currently just 2 classes allowed
    TODO:
        Make this function more readable/efficient
        1) rprob doesn't need concatenation
        2) all_scores can be 2 vars instead
        3) variable names
        4) I think po and pf are just linear interpolation across 0 to 1
    ARGUMENTS:
        y_obs: array[n] of classes observed
                These need to start at index 1, and be integers
        probabilities: array[n,k] of probabilities to be in class (1 to k) for n observations
            NB: predictions can be inferred from this, but aren't necessary here
    """
    # used by old method...
    all_scores = {
        'prob_below':probabilities[:,0],
        'prob_above':probabilities[:,1],
        'class':y_obs,
        }
    # how many columns in proba matrix
    nclasses=np.shape(probabilities)[1] 
    # how many samples
    nt=len(y_obs) 
    
    # rprob is 2*nt long (prob below,prob above) just 1 dimensional now..
    rprob=np.concatenate((all_scores['prob_below'], all_scores['prob_above']))
    iobs = all_scores['class']

    #Fill up scoring table
    p1 = np.array([0,1/nclasses])
    p2 = np.array([1/nclasses,1])
    dp = p2-p1
    # TODO: Why make this one bigger than necessary in both dims?
    score = np.zeros((nclasses,nclasses))
    for igf in range(nclasses):
        for igo in range(nclasses):
            as_ = 0.
            n  = 0.

            for idum1 in range(51):
                po = p1[igo]+dp[igo]*(idum1)/50.
                for idum2 in range(51):
                    pf = p1[igf]+dp[igf]*(idum2)/50.
                    s = 3.0*(1.-np.abs(pf - po)+pf**2-pf+po**2-po)-1.
                    as_ = as_+s
                    n = n + 1
            score[igf,igo] = as_/float(n)

    sa1w = 0.
    sa1b = 0.
    aleps = 0.
    leps = np.zeros(nt)
    for it in range(nt):
        s1 = 0.
        sw = 99999
        sb = -99999

        for igf in range(nclasses):
            sb = max(sb,score[igf,iobs[it]-1])
            sw = min(sw,score[igf,iobs[it]-1])
            s1 = s1 + rprob[int(it + (igf)*nt)]*score[igf,iobs[it]-1]

        sa1w = sa1w + sw
        sa1b = sa1b + sb
        aleps = aleps + s1

        if abs(s1) < 1e-5:
            leps[it] = 0.
        elif s1 > 0.:
            leps[it] = 100*s1/sb
        else:
            leps[it] = -100*s1/sw

        
        if aleps > 0.:
            if abs(sa1b) > 1e-3:
                aleps = 100*aleps/sa1b
            else:
                aleps = 0.0
        else:
            if abs(sa1b) > 1e-3:
                aleps = -100*aleps/sa1w
            else:
                aleps = 0.0
                
    return leps

# def combo_method_SKL_validation(X1,X2,Y, years, 
#                                 cv=LeaveOneOut(),
#                                 target='AR',
#                                 ):
#     """
#         Cross validate the bom combo method
#         ARGS:
#             DF: dataframe with nino,soi,region columns
#             years: list of years to validate (or range or ndarray)
#             cv: cross validator from sklearn with inherited "split" method
#                 default = LeaveOneOut()
#     """
#     if cv is None:
#         cv=LeaveOneOut()

#     i_sub = X1.index.year.isin(years)
#     X1_sub, X2_sub, Y_sub = X1[i_sub], X2[i_sub], Y[i_sub]
#     if not isinstance(years,np.ndarray):
#         years = np.array(years)
    
#     scores=[]
#     i=0
#     ## Iterate over dataset using cross validation arg
#     for i_train, i_test in cv.split(X1_sub):
#         # training and verification years
#         years_tr = years[i_train]
#         years_ver= years[i_test]
        
#         # probability from combo method
#         i_prob = combo_method_SKL(
#             DF=DF_sub,
#             training_range=years_tr,
#             prediction_range=years_ver,
#             show_scatter=False,plt=None,print_leps_skill=False)
        
#         ## verification based on median of training data
#         # use training data to set our median TR count
#         obs_tr = DF_sub[target][i_train]
#         obs_ver = DF_sub[target][i_test]
#         median = obs_tr.median()
#         # obs over verification space using that median
#         y_ver = np.ones(len(years_ver),dtype=int)
#         y_ver[obs_ver.values < median] = 1
#         y_ver[obs_ver.values >= median] = 2
#         # scores from probability vs observation
#         scores.append(get_classification_scores(i_prob,y_ver))
#         # add some meta data for easier analysis
#         scores[i]['years_training']=years_tr
#         scores[i]['years_verification']=years_ver
#         #scores[i][]
#         i=i+1
#     # Calculate some averages
#     # validation_means={"LEPS":0, }
#     # for score in scores:
#     #     mean_leps = mean_leps+score['LEPS skill']
#     return scores

def get_classification_scores(y_prob,y_obs):
    """
    ARGS:
        y_prob: probabilities of class numpy array [n,2]: [[class1, class2],...]
        y_class: true class: [1,2,1,...]
    """
    scores={}
    
    # prob[:,0] is probabilities of class 1, or probability of less than median TC count
    # hits: correctly predict greater than median TC count
    hits = np.sum((y_prob[:,0]<0.5) * (y_obs==2))
    # misses: incorrectly predict less than median TC count
    misses = np.sum((y_prob[:,0]>=0.5) * (y_obs==2))
    # false alarm: incorrectly predict greater than median TC count
    false_alarms = np.sum((y_prob[:,0]<0.5) * (y_obs==1))
    # correct low prediction
    hits2 = np.sum((y_prob[:,0]>=0.5) * (y_obs==1))

    # Contingency table: predictions is y axis, obs is x axis
    scores['contingency'] = "%2d,%2d;%2d,%2d"%(hits2,misses,false_alarms,hits)
    scores['class'] = y_obs
    scores['prob_below']= y_prob[:,0]
    scores['hit1']=hits2
    scores['hit2']=hits
    scores['miss1']=false_alarms # missed class 1
    scores['miss2']=misses # missed class 2
    #print("DEBUG: y_test, y_prob",y_test.values,y_prob)
    
    ## bias = class 2 predictions / (class 2 occurrences)
    # bias is 0 to infinity, ignore divide by zero warning
    with np.errstate(divide='ignore',invalid="ignore"):
        scores['bias'] = (hits+false_alarms)/(hits+misses)
    # accuracy is 0 to 1
    scores['accuracy']=(hits+hits2)/len(y_obs)
    
    LEPS_test=calculate_classed_leps_skill(y_obs,y_prob)
    scores['LEPS skill']=(LEPS_test)
    scores['BSS'] = brier_skill_score(y_obs,y_prob[:,0])
    #t1,t2,t3 = get_terciles(y_prob)
    #scores['tercile_1']=t1
    #scores['tercile_2']=t2
    #scores['tercile_3']=t3

    ## Weighted percent correct
    ## 
    scores['WPC'] = "not yet implemented"
    ## Relative Operating Characteristic
    ##     https://cawcr.gov.au/projects/verification/#ROC
    scores['ROC']= "not yet implemented"
    
    return scores 

def lda_compute_forecast_probs(subset,var,within_class_scatter_matrix,class_feature_means,nclasses,predictor,year):
   """
   Compute forecast probs
   """
   norm_factor = subset.shape[0]-nclasses
   xconv = 1/(within_class_scatter_matrix[0]/norm_factor)

   #Calculate group weightings
   g2 = np.zeros([nclasses])
   g2[:] = -2.0*np.log(1/float(nclasses))

   #Append aposterior probabilities for this year
   aprob_dict = {'year':year,'var':var,'prob':1.0/float(nclasses)}
    
   #Calculated estimated unconditional destiny
   diff = np.zeros([nclasses])
   for i,c in enumerate(class_feature_means.columns):
        diff[i] = predictor  -  class_feature_means[c][var]

   #distg = a*bT *a
   dist = diff*xconv*diff

   sum_exp = 0
   for d in dist:
        sum_exp += np.exp(-0.5*d)

   #Calculate posterior probabilities
   prob_dict = []
   probs=[]
   for d in dist:
        prob_dict.append({'year':year,'var':var,'prob': np.exp(-0.5*d)/sum_exp })
        probs.append(np.exp(-0.5*d)/sum_exp)
        
   return probs

def lda_forecast_jessewrap(DF, training_range, prediction_year,
                           var='nino3.4 anom 3m mean',
                           target='AR'
                           ):
    
    start = dt.datetime(training_range[0],9,1)
    end = dt.datetime(prediction_year,9,1)

    ## I'm using rolling 3m mean, so use month 9 not month 7
    # for JAS means
    mx = DF.index.month == 9
    july_obs = DF[mx][start:end].copy()
    
    ## Training and prediction ranges as list of indices
    i_tra = DF.index.year.isin(training_range)
    # true where year == training range years
    
    ## X and y from data frame
    subset = DF[[var,target]].copy()[i_tra]
    
    # y is based on median: 1 for less or equal, 2 for more
    median = DF[target][i_tra].median()
    # add 'class' column based on training median
    subset.loc[:,('class')] = subset.apply(lambda row : prediction.classify_median(row,target,median),axis=1).values
    
    
    #For %chance above median, nclasses=2
    nclasses=2
    temp_df = []

    # LDA internal scatter matrix
    within_class_scatter_matrix, class_feature_means = lda_generate_scatter_matrix(subset[[var,'class']])
    # LDA predictor based on var argument
    yx = july_obs.index.year != prediction_year
    predictor = july_obs[~yx][var].values
    # actual LDA probabilities of above/below median
    probs = lda_compute_forecast_probs(subset[[var,'class']],var,within_class_scatter_matrix,class_feature_means,nclasses,predictor,prediction_year)
    probs.append(prediction_year)
    #Compute class scores
    # class column using full range of given years
    july_obs.loc[:,('class')] = july_obs.apply(lambda row : prediction.classify_median(row,target,median),axis=1).values
    score = july_obs[~yx]['class'].values[0]
    probs.append(score)
    # Put probabilities, year, and class into a dataframe with titles
    temp_df.append(pd.DataFrame([probs],columns=['prob_below','prob_above','year','class']).set_index('year'))
    all_scores=pd.concat(temp_df)
    
    return all_scores

def lda_generate_scatter_matrix(subset):
   """
   Create the within class scatter matrix
   """
   #dim is number of subset columns - class column
   dim = subset.shape[1]-1
   
   class_feature_means =  pd.DataFrame()
   for c, rows in subset.groupby('class'):
        class_feature_means[c] = rows.mean()

   within_class_scatter_matrix = np.zeros((dim,dim))
   for c, rows in subset.groupby('class'):
        rows = rows.drop(['class'], axis=1)
        s = np.zeros((dim,dim))
        for index, row in rows.iterrows():
            x, mc = row.values.reshape(dim,1), class_feature_means[c].drop(index='class').values.reshape(dim,1)        
            s += (x - mc).dot((x - mc).T)    
        within_class_scatter_matrix += s

   return within_class_scatter_matrix,class_feature_means

def prior_forecasts(year):
    """
    Read prior forecasting results from csv output
    """
    url = 'data/%d-%d/TC_output.csv'%(year,year%100+1)
    names="regionstr","metricstr","LEPS","p1","p2","t1","t2","t3"
    df=pd.read_csv(url, header=None, names=names,index_col=False)
    # map region string to region
    regionmap={
        "OZ105_130_25S.data":"AR-NW",
        "OZ125_1425.data":"AR-N",
        "OZ90_125.data":"AR-W",
        "OZ1425_160.data":"AR-E",
        "OZ90_160.data":"AR",
        "SPO1425_165.data":"SPO-W",
        "SPO165_240.data":"SPO-E",
        "SPO1425_240.data":"SPO",
        }
    metricmap={"ERSSTv5_NINO_34.txt":"nino","SOI_BoM.txt":"SOI",}
    # Create forecast dictionary for testing
    ret={}
    for index, row in df.iterrows():
        region=regionmap[row["regionstr"].strip()]
        metric=metricmap[row["metricstr"].strip()]
        ret[region+"_"+metric]=row["p1"]
    
    return ret

def read_all(
    url_local = 'all_data.csv',
    refresh=False,
    url_sam = 'data/sam_1951_2021.csv',
    url_esam = 'data/esam_1949_2022.csv',
    TC_folder = 'data/2021-22/',
    ):
    """
    Wrapper for reading all input and target data
    Joined together into one dataframe on 'date'
    Adds "season" column:  1,1,2,2,2,3,3,3,4,4,4,1 based on month

    Saves all info into all_data.csv
    if refresh is True, download the data from the urls again and remake the csv
        
    """
    
    if not refresh and os.path.isfile(url_local):
        all_data = pd.read_csv(url_local,parse_dates=True)
        all_data['date'] = pd.to_datetime(all_data['date'])
        all_data.set_index(all_data.date,inplace=True)
        # make sure date is index
        return all_data

    #dmi_data = read_dmi(dmi_url)
    nino_data = prediction.read_nino()
    soi_data = prediction.read_soi()
    sam_data = read_sam(url_sam)
    esoi_data = read_equatorial_soi(url_esam)
    iod_data = read_iod()

    # read idck TC counts in AR and sub-regions
    # columns: [index:'date', 'idck_AR', 'idck_AR-W', ...
    idck_tc_data = read_tc_from_idckmstm0s()
    tc_data = prediction.read_processed_tc(TC_folder)

    ## combine these datasets into one dataframe
    ## how='outer' means keep all info, fill in missing columns with nans
    ## join using index columns (make sure read_data returns datetime index column)
    merge_args = {"left_index":True,"right_index":True,"how":'outer'}
    tmp_data1 = pd.merge(nino_data,soi_data,**merge_args)
    tmp_data2 = pd.merge(tmp_data1,iod_data[['IODE','IODW','IODE anom','IODW anom','DMI','DMI anom']],**merge_args)
    tmp_data3 = pd.merge(tmp_data2,sam_data['SAM'],**merge_args)
    tmp_data4 = pd.merge(tmp_data3,esoi_data['ESOI'],**merge_args)
    
    # join TC
    tmp_data5 = pd.merge(tmp_data4,idck_tc_data,**merge_args) 
    all_data = pd.merge(tmp_data5,tc_data,**merge_args)

    # Add Trans-Nino Index (nino4 anom - nino1+2 anom)
    all_data.insert(2,"TNI", all_data['nino4 anom'].values-all_data['nino1+2 anom'].values,allow_duplicates=True)

    #Compute 3-month forward rolling means for SOI and Nino3.4
    #by default pandas uses a trailing mean (2 prior months+current)/3.0
    for column in [
        'nino1+2', 'nino1+2 anom',
        'nino3', 'nino3 anom',
        'nino4', 'nino4 anom',
        'nino3.4', 'nino3.4 anom',
        'SOI',
        'DMI', 'DMI anom',
        'IODE', 'IODE anom',
        'IODW', 'IODW anom',
        'TNI',
        'SAM',
        'ESOI',
        ]:
        # take 3m rolling mean as new column (column index = 2, doesn't matter)
        all_data.insert(2,column+' 3m mean', all_data[column].rolling(window=3).mean(), allow_duplicates=True)
    
    # Add season column
    all_data.insert(2,"season",all_data.index.month%12//3 + 1, allow_duplicates=True)
    # print([(i+1)%12//3+1 for i in range(12)]) # 1,1,2,2,2,3,3,3,4,4,4,1
    # Summer = 1, Autumn = 2, Winter = 3, Spring = 4
    
    # Also add year
    all_data.insert(2,'year', all_data.index.year.values, allow_duplicates=True)

    # Save to file 
    all_data.to_csv(url_local,index_label='date')
    return all_data

def read_dmi():
    """
    Read Indian Ocean Dipole Mode Index data from NOAA CPC
        'https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'
    """
    url='https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'

    # Allow vectorised string addition
    def add_strings(a,b):
        return a+b

    #Skip header - we will specify our own
    columns=['year','Jan', 'Feb', 'Mar' ,'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep' , 'Oct' , 'Nov', 'Dec']
    dmi_data = pd.read_csv(url,skiprows=1,names=columns,delim_whitespace=True)
    #Cull non-numeric data 
    nx = dmi_data.year.str.isnumeric()
    dmi_data = dmi_data[nx]
    dmi_data.set_index(dmi_data.year,inplace=True)
    dmi_data.drop('year',axis=1,inplace=True)
    dmi_data.columns=list(range(1,13))

    # Now stack dataframe to convert from rows to a column
    stack=dmi_data.stack().to_frame()
    stack.columns=['DMI']
    stack.reset_index(inplace=True)
    # make datetime column, make it index
    stack['date']=pd.to_datetime(add_strings(stack['year'].map(str),stack['level_1'].map(str)),format='%Y%m')
    stack.set_index('date',inplace=True)
    stack.sort_index(inplace=True)

    # Remove year, month columns (they are part of index column anyway)
    stack.drop(columns=['year', 'level_1'],inplace=True)
    # convert string to float
    stack['DMI'] = pd.to_numeric(stack['DMI'],errors='coerce')
    # replace -9999 with NaN
    stack.replace({-9999.0: np.NaN,},inplace=True)

    # return dataframe with 'DMI' column
    return stack

def read_equatorial_soi(url="https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"):
    """ read https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for """
    # columns are months, rows are years, first column is year
    names=['year','1','2','3','4','5','6','7','8','9','10','11','12']
    
    # No header
    df = pd.read_csv(url, delim_whitespace=True, names=names)
    
    # remove rows that begin don't begin with yyyy
    df = df[df['year'].apply(lambda x: str(x).isdigit())]
    
    # replace strings with floats, coerce failures into NaN
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') 
    
    ## NOW change month columns to rows
    df2=df.melt(id_vars=['year'],value_vars=names[1:],var_name='month')

    # make a datetime column
    df2['date'] = pd.to_datetime(df2[['year','month']].assign(DAY=1))
    # Sort and index using date column
    df2.set_index('date',inplace=True)
    df2.sort_index(inplace=True)
    
    # rename 'value' to ESOI
    df2.rename({'value':'ESOI'},axis='columns',inplace=True)

    # replace 999.9 with NaN
    df2['ESOI'] = df2['ESOI'].replace(999.9,np.NaN)

    return df2

def read_iod():
    """ Japan Met Agency has IOD stuff available, including DMI """
    __url_jma_base__='https://ds.data.jma.go.jp/tcc/tcc/products/elnino/index/sstindex/base_period_9120/'
    __url_iod_jma__={'IODW':__url_jma_base__+'WIN/sst',
                    'IODW anom':__url_jma_base__+'WIN/anomaly',
                    'IODE':__url_jma_base__+'EIN/sst',
                    'IODE anom':__url_jma_base__+'EIN/anomaly',
                    'DMI':__url_jma_base__+'DMI/sst',
                    'DMI anom':__url_jma_base__+'DMI/anomaly',
    }
    urls=__url_iod_jma__
    # columns are months, rows are years, first column is year
    names=['year','1','2','3','4','5','6','7','8','9','10','11','12']
    
    df_all = None
    
    for name,url in urls.items():
        ## TODO: make reading this format into a function.
        # No header
        df = pd.read_csv(url, delim_whitespace=True, names=names)

        # remove rows that begin don't begin with yyyy
        df = df[df['year'].apply(lambda x: str(x).isdigit())]

        # replace strings with floats, coerce failures into NaN
        cols = df.columns
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') 

        ## NOW change month columns to rows
        df2=df.melt(id_vars=['year'],value_vars=names[1:],var_name='month')

        # make a datetime column
        df2['date'] = pd.to_datetime(df2[['year','month']].assign(DAY=1))
        # Sort and index using date column
        df2.set_index('date',inplace=True)
        df2.sort_index(inplace=True)

        # rename 'value' to ESOI
        df2.rename({'value':name},axis='columns',inplace=True)

        # replace 99.90 with NaN
        df2[name] = df2[name].replace(99.90,np.NaN)
        #print(name,)
        #print(df2)
        
        if df_all is None:
            df_all = pd.DataFrame(df2[name])
        else:
            df_all = df_all.join(df2[name].copy(),how='outer')
        
    return df_all

def read_processed_data(
    url_nino='data/2021-22/ERSSTv5_NINO_34.txt',
    url_SOI='data/2021-22/SOI_BoM.txt',
    TC_folder='data/2021-22/',
    ):

    #these are yyyy m float float float (12 floats, all the same value)
    excess = ['2','3','4','5','6','7','8','9','10','11','12']
    names_nino=['year','month', 'nino3.4 anom',]+excess
    names_SOI=['year','month', 'SOI',]+excess
    
    # No header line or commas
    df_nino = pd.read_csv(url_nino, header=None, delim_whitespace=True, names=names_nino)
    df_nino.drop(excess, axis='columns', inplace=True)
    df_SOI  = pd.read_csv(url_SOI,  header=None, delim_whitespace=True, names=names_SOI)
    df_SOI.drop(excess, axis='columns', inplace=True)
    
    for df in [df_nino,df_SOI]:
        cols = df.columns
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # make a datetime column
        df['date'] = pd.to_datetime(df[['year','month']].assign(DAY=1))
        
        # drop year/month
        df.drop(['year','month'],axis='columns',inplace=True)
        
        # set index
        df.set_index('date',inplace=True)

    # lets add TC dataset
    #tc_data = read_tc_database(__url_tc__)
    tc_data = prediction.read_processed_tc(TC_folder)

    # combine dataframes
    tmp1 = pd.merge(df_nino, df_SOI, right_index=True, left_index=True, how='outer')
    df = pd.merge(tmp1,tc_data,left_index=True,right_index=True,how='outer')
    for column in ['nino3.4 anom', 'SOI']:
        # take 3m rolling mean as new column (column index = 2, doesn't matter)
        df.insert(2, column+' 3m mean', df[column].rolling(window=3).mean(), allow_duplicates=True)
    
    return df

def read_sam(url = 'data/sam_1951_2021.csv'):
    """
    Read SAM from a csv or NOAA websie
        # monthly SAM from 1948->2011
        SAM_url="https://psl.noaa.gov/data/correlation/sam.20crv2c.short.data"
        # data up to 2021 using provided ncl code from https://psl.noaa.gov/data/20thC_Rean/timeseries/monthly/SAM/
        SAM_url="sam_1951_2021.csv"
    RETURNS
        dataframe with monthly SAM, datetime indexed
    """
    
    # columns are months, rows are years, first column is year
    names=['year','1','2','3','4','5','6','7','8','9','10','11','12']
    SAM_in = pd.read_csv(url,names=names,delim_whitespace=True)
    
    # trailing rows may be comments if reading from noaa site, drop them from table
    SAM_in = SAM_in[SAM_in['year'].apply(lambda x: str(x).isdigit())]
    # remove nan values (result from header if reading from noaa)
    SAM_in.dropna(inplace=True)
    
    # change month columns to rows
    SAM_in2=SAM_in.melt(id_vars=['year'],value_vars=names[1:],var_name='month')

    # make a datetime column
    SAM_in2['date'] = pd.to_datetime(SAM_in2[['year','month']].assign(DAY=1))
    # Sort and index using date column
    SAM_in2.set_index('date',inplace=True)
    SAM_in2.sort_index(inplace=True)
    # remove superfluous year/month columns
    SAM_in2.drop(['year','month'],axis='columns',inplace=True)
    # finaly make sure index is float
    SAM_in2['SAM'] = SAM_in2['value'].astype(float)
    SAM_in2 = SAM_in2.drop(['value'],axis='columns')

    return SAM_in2

def read_tc_from_idckmstm0s(
    csv_url="http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
    start=None,
    end=None,
    remove_types="LOU"):
    """
    Read historical TC .csv database: IDCKMSTM0S.csv
    from http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv
        file based on operational analysis of each event best tracks
        noname entries:
            either Lows that didn't quite make TC intensity
            or else they were post event analysed as being TC (but not named in operation)
        The 'type' column ('T','D',or 'L'):
            T: TC
            E: extra-tropical cyclone or east coast low
            D: Draft: awaiting completion of best track
            L: tropical low failed to attain TC strength
            O: other
            U: unknown
            I will filter out LOU types
        details here: http://www.bom.gov.au/cyclone/history/database/TC_Database_Structure_Oct2011.pdf
    """
    columns=['NAME','DISTURBANCE_ID','TM','TYPE','DATA_SRC','SURFACE_CODE','CYC_TYPE','LAT','LON','POSITION_METHOD','POSITION_UNCERTAINTY','DVORAK_DATA_T_NO','DVORAK_MODEL_T_NO','DVORAK_PATTERN_T_NO','DVORAK_FINAL_T_NO','DVORAK_CI_NO','CENTRAL_PRES','CENTRAL_PRES_UNCERTAINTY','CENTRAL_PRES_METHOD','PRES_WIND_RELATION_USED','ENV_PRES','ENV_PRES_UNCERTAINTY','MN_RADIUS_OUTER_ISOBAR','MN_RAD_OUT_ISOBAR_UNCERTAINTY','MN_RADIUS_GF_WIND','MN_RADIUS_GF_SECNE','MN_RADIUS_GF_SECSE','MN_RADIUS_GF_SECSW','MN_RADIUS_GF_SECNW','MN_RADIUS_SF_WIND','MN_RADIUS_SF_SECNE','MN_RADIUS_SF_SECSE','MN_RADIUS_SF_SECSW','MN_RADIUS_SF_SECNW','MN_RADIUS_HF_WIND','MN_RADIUS_HF_SECNE','MN_RADIUS_HF_SECSE','MN_RADIUS_HF_SECSW','MN_RADIUS_HF_SECNW','MN_RADIUS_MAX_WIND','MN_RADIUS_MAX_WIND_UNCERTAINTY','MN_RADIUS_GF_WIND_UNCERTAINTY','MN_RADIUS_SF_WIND_UNCERTAINTY','MN_RADIUS_HF_WIND_UNCERTAINTY','MN_RADIUS_MAX_WIND_METHOD','MN_RADIUS_GF_WIND_METHOD','MN_RADIUS_SF_WIND_METHOD','MN_RADIUS_HF_WIND_METHOD','WIND_SPD_PER','MAX_WIND_SPD','MAX_WIND_SPD_UNCERTAINTY','MAX_WIND_SPD_METHOD','MAX_WIND_GUST_PER','MAX_WIND_GUST','MAX_WIND_GUST_METHOD','MN_EYE_RAD','MN_EYE_RAD_UNCERTAINTY','MN_EYE_RAD_METHOD','MAX_REP_WIND_SPD','MAX_REP_WIND_DIR','MAX_REP_WIND_METHOD','MAX_REP_WIND_LON','MAX_REP_WIND_LAT','MAX_REP_WAV_HT','MAX_REP_WAV_METHOD','MAX_REP_WAV_LON','MAX_REP_WAV_LAT','MAX_REP_SWL_HT','MAX_REP_SWL_DIR','MAX_REP_SWL_PER','MAX_REP_SWL_METHOD','MAX_REP_SWL_LON','MAX_REP_SWL_LAT','MAX_REP_TIDE_ANOM','MAX_REP_TIDE_ANOM_UNCERTAINTY','MAX_REP_TIDE_ANOM_METHOD','MAX_REP_TIDE_ANOM_LON','MAX_REP_TIDE_ANOM_LAT','COMMENT']
    
    tc_data = pd.read_csv(csv_url,skiprows=3000,names=columns) #Skip rows to skip header and data before ~1964

    #Remove rows with no timestamp - caused by overflowing comments proceeding onto subsequent lines
    for i,r in tc_data.iterrows():
        if not re.search(r'\d+',r['TM']): #There are no digit characters in the time field
            #print (i,r)
            tc_data.drop(i,inplace=True) #Drop the row from the DataFrame
            
    tc_data['date']=pd.to_datetime(tc_data['TM'],format="%Y-%m-%d %H:%M")
    tc_data.set_index('date',inplace=True)
    tc_data.sort_index()

    #Convert longitude fields to floats
    tc_data.loc[:,'LON']=pd.to_numeric(tc_data['LON'])
    
    # remove Tropical Lows (these are not TCs), Other and Unknown types too
    for ch in remove_types:
        tc_data = tc_data[tc_data["TYPE"].str.contains(ch)==False].copy()

    #Now create dictionary defining the longitude limits of each region
    regions = { 'idck_AR': [90,160], 'idck_AR-W': [90,125], 'idck_AR-N': [125,145], 'idck_AR-E': [145,160], 'idck_AR-NW' : [105,130]}
    all_storm_counts = [] #Initialise list of dataframes
    ## SPO region does not appear to be within IDCK csv file
    
    #Loop over hindcast period to produce data frame of storm counts in each reagions
    if start is None:
        start = dt.datetime(1960,1,1)
    if end is None:
        end = dt.date.today()
    
    for year in range(start.year,end.year):
        # indices for each july-june year/season
        tx=(tc_data.index >= dt.datetime(year,7,1)) & (tc_data.index <=dt.datetime (year+1,6,30))  
        subset=tc_data[tx]
        d = {} #Temporary dictionary to hold a seasons counts
        # one entry per year, make it september to match rolling means for prediction
        d['date'] = dt.datetime(year,9,1) 
        
        for r in regions.keys():
            rx = (subset.LON >= regions[r][0]) &  (subset.LON <= regions[r][1])
            names = subset[rx].NAME.unique()
            types = subset[rx]['TYPE'].unique()
            counts = len(names)
            if False:
                print("region:%s, year:%d"%(r,year))
                print(names)
                print(types)
                print("counts: %d"%counts)
            d[r] = counts
        
        all_storm_counts.append(pd.DataFrame(d,index=[year]))
    
    #Concatenate
    storm_counts = pd.concat(all_storm_counts)
    storm_counts.set_index('date',inplace=True)

    return storm_counts

def test_prediction(
    prediction_year,
    start_year=1970,
    print_checks=True,
    assert_checks=True):
    """
    Compare new python code to results from 2020 or 2021
        comparison probabilities are in TC_output.csv within TC_folder
        2021:
            AR 2021/2022 http://www.bom.gov.au/climate/cyclones/australia/archive/20211011.archive.shtml
            SPO 2021/2022 http://www.bom.gov.au/climate/cyclones/south-pacific/archive/20211012.archive.shtml
        
    """
    #prior_results = Forecast[prediction_year]
    prior_results = prior_forecasts(prediction_year)
    folder="data/%d-%d/"%(prediction_year,prediction_year%100+1)

    ## read nino, soi used in 2020/21 forecast
    data = read_processed_data(
        url_nino=folder+'ERSSTv5_NINO_34.txt',
        url_SOI=folder+'SOI_BoM.txt',
        TC_folder=folder,
    )
    
    # just september values
    sep = (data.index.month == 9)
    dt1,dt2 = dt.datetime(start_year,9,1),dt.datetime(prediction_year,9,1)
    training_range = range(start_year,prediction_year)
    data = data[sep].copy()
    data = data[dt1:dt2].copy()    
    
    # run model on dataset
    targets=['AR','AR-W','AR-N','AR-NW','AR-E','SPO','SPO-E','SPO-W']
    print("inputs:... from year %d with prediction year = %d"%(start_year,prediction_year))
    print(data[['nino3.4 anom 3m mean','SOI 3m mean']+targets][-3:])
    X1 = data['nino3.4 anom 3m mean']
    X2 = data['SOI 3m mean']
    # outcome lists for display
    n1,n2,n3 = [],[],[]
    s1,s2,s3 = [],[],[]

    for target in targets:
        Y=data[target]
        prob_nino, prob_soi  = prediction.combo_method_SKL(
            X1=X1,
            X2=X2,
            Y=Y,
            prediction_year=prediction_year,
            )    
    
        paulnino = lda_forecast_jessewrap(DF=data,
                                  training_range=training_range,
                                  prediction_year=prediction_year,
                                  target=target,
                                  var='nino3.4 anom 3m mean',
                                 )
        paulsoi   = lda_forecast_jessewrap(DF=data,
                                  training_range=training_range,
                                  prediction_year=prediction_year,
                                  target=target,
                                  var='SOI 3m mean',
                                 )
        
        # Save results
        n1.append(prob_nino[0][0])
        n2.append(prior_results[target+"_nino"])
        n3.append(paulnino['prob_below'].values[0])
        s1.append(prob_soi[0][0])
        s2.append(prior_results[target+"_SOI"])
        s3.append(paulsoi['prob_below'].values[0])
        
    dfdict={
        'region':targets,
        'nino_python':n1,
        'nino_fortran':n2,
        'nino_paul':n3,
        'SOI_python':s1,
        'SOI_fortran':s2,
        'SOI_paul':s3,
    }
    df = pd.DataFrame(dfdict) 
    if print_checks:
        print(df[['region','nino_python','nino_fortran','SOI_python','SOI_fortran']])
    
    if assert_checks:
        nino_diffs = df['nino_python'].values - df['nino_fortran'].values
        assert np.all(np.abs(nino_diffs) <= 0.0005), "There is a difference in nino based LDA"
        soi_diffs = df['SOI_python'].values - df['SOI_fortran'].values
        assert np.all(np.abs(soi_diffs) <= 0.0005), "There is a difference in SOI based LDA"
        print("INFO: ASSERTION TESTS PASSED")
    
