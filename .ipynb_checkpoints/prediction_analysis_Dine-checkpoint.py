import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os, re
import datetime as dt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit


# Modified combo_method_SKL for tercile LDA + allow generalised input for training_years and prediction_years

def combo_method_SKL_Dine(X1,X2,Y,
                     training_years=None,
                     prediction_years=None,
                     ):
    """
        ARGS:
            X1,X2: input pandas Series for LDA1 and LDA2 
                these are expected to be nino3.4 anom 3m mean and relative nino3.4 anom 3m mean
            Y: pandas Series with mean SON rainfall over MDB and datetime index
            training_years=None : if not set assumed to be (1900, prediction_years) 
            prediction_years=None : if not set assumed to be final year
            
        Returns:
            prediction_probabilities: numpy array[n_predictions,n_classes]
                row = predicted probabilities for each year in the 
                prediction range being below the median (class0)
                around median (class1)
                or above the median (class2)
    """
    ## Training and prediction ranges as list of indices
    if prediction_years is None:
        prediction_years = X1.index.year.values[-1]
    # training_range = range(1900,prediction_year)
    if training_years is None:
        training_years = range(1900,prediction_years[0])
    
    i_tra = X1.index.year.isin(training_years)
    i_tra_Y = Y.index.isin(training_years)
    i_ver = X1.index.year.isin(prediction_years)
    # i_ver looks like [False,False,...,True] 
    # true where year == training range years


    ## X and y from data frame
    X_roni= X1.copy()[i_tra]
    X_roni_ver = X1.copy()[i_ver]
    X_oni = X2.copy()[i_tra]
    X_oni_ver = X2.copy()[i_ver]
    
    
    # y is based on median: 0 for less, 1 for around, 2 for over 
    y0 = Y[i_tra_Y]
    q33, q67 = np.quantile(y0, [1/3, 2/3])
    y = np.ones(len(y0),dtype=int)
    y[y0.values < q33] = 0
    # y[(y0.values >= q33) & (y0.values <= q67)] = 1
    y[y0.values > q67] = 2
    
    ## Initialise LDA models
    LDA_roni = LinearDiscriminantAnalysis(solver='svd',priors=[1/3,1/3,1/3])
    LDA_oni = LinearDiscriminantAnalysis(solver='svd',priors=[1/3,1/3,1/3])
    
    ## fit models using training data
    LDA_roni.fit(X_roni.values.reshape(-1, 1), y)
    LDA_oni.fit(X_oni.values.reshape(-1, 1), y)
    
    ## Predict probability of class0,class1,class2 for prediction input sets
    # reshape required as we are only using 1 input column
    prob_roni = LDA_roni.predict_proba(X_roni_ver.values.reshape(-1,1))
    prob_oni = LDA_oni.predict_proba(X_oni_ver.values.reshape(-1,1))

    # probabilities are given as array[n_predictions,n_classes]
    # so 3 columns, one for class 0, second for class 1, third for class 2
    
    return prob_roni, prob_oni, q33, q67



# Define BSS & LEPS & Hit/miss (contingency table, accuracy, bias)

def brier_skill_score(obs,proba, 
    assumed_BS_ref_prob=1/3, show_calculations=False,
    ):
    """ 
        Brier Skill is essentially mean square error:
            (prediction proba - [0 if true, 1 if false])**2
        BSS = 1 - BS/BS_ref
        BS_ref = score of reference method we are trying to beat
            in this case BS_ref is an assumed 33.333% for each tercile
        
        BSS = 1 the forecast has perfect skill compared to climatology;
        BSS = 0 the forecast has no skill compared to climatology);
        BSS = a negative value the forecast is less accurate than climatology.
        
        ARGUMENTS:
            proba is the probability distribution
            obs is list of class 0, 1 or class 2 occurrences
            assumed_... is reference climatological model estimate(s) of class 0,1,2
                this could be updated to (for example) the number of class 1 outcomes / total number of predictions
                currently is set to 1/3% (expect 1/3 to be greater than median, 1/3 to be =, 1/3 to be below)
    """
    # if proba is [n rows, 3 columns] then it is probably [prob_below;prob_middle;prob_above]
    # fix it with warning
    if len(np.shape(proba)) == 2 and np.shape(proba)[1]==3: # if 2D array (n_years,tercile) and 3 columns
        # print("WARNING: brier_skill_score proba is assuming the first column to be probability of class 0")
        t1 = proba[:,0]
        t2 = proba[:,1]
        t3 = proba[:,2]
        

    # remove any missing values
    df = pd.DataFrame({'obs':obs,'t1':t1, 't2':t2, 't3':t3})
    df = df.dropna()
    
    # Brier Score: 0 if 100% prediction is correct, 1 if 100% prediction is wrong
    df['BS'] = (
        (df['t1'] - (df['obs']==0).astype(int)) **2 + # If t1 is assigned 1.00 at lower tercile (obs), then 1-1 = 0 => perfect prediction  
        (df['t2'] - (df['obs']==1).astype(int)) **2 + 
        (df['t3'] - (df['obs']==2).astype(int)) **2
    )/3
    df['BS_ref'] = (
        (assumed_BS_ref_prob - (df['obs']==0).astype(int)) **2 + 
        (assumed_BS_ref_prob - (df['obs']==1).astype(int)) **2 + 
        (assumed_BS_ref_prob - (df['obs']==2).astype(int)) **2
    )/3
    df['BSS'] = 1 - df['BS']/df['BS_ref']
    # if show_calculations:
        # print(df)
    # return mean of BSS
    BSS = df['BSS'].mean()

    return BSS


#############################################
# rows = forecast tercile (p1, p2, p3), columns = observed tercile
M = np.array([[8, -1, -7],
              [-1, 2, -1],
              [-7, -1, 8]]) / 27.0
U = M.max(axis=0)          # best possible score per observed tercile: 8/27, 2/27, 8/27
L = M.min(axis=0)          # worst possible score per observed tercile: -7/27, -1/27, -7/27

def leps_skill(j, P): # switched to match my code 
    """Correct LEPS skill."""
    s = (P * M[:, j].T).sum(axis=1)       # s_i = sum_k p_k M[k, j_i]
    u = U[j]
    l = L[j]
    S = s.sum()
    denom = u.sum() if S >= 0 else abs(l.sum())
    #return {"s": s, "mean_S": s.mean(), "sum_S": S, "sum_S_m": denom, "LEPS_skill": S / denom}
    leps_skill = S / denom
    return leps_skill
###################################################################
    

def calculate_classed_leps_skill(y_obs, probabilities):
    """
 
    ARGUMENTS:
        y_obs: array[n] of classes observed
        probabilities: array[n,3] of probabilities to be in class (0 to 2) for n observations
            NB: predictions can be inferred from this, but aren't necessary here

    Added by DY: 
    Tercile (three-category) LEPS described in Fawcett et al., 2005. 
    Suppose the probability of a tercile 1 outcome at a given point is given as p1, 
    with the corresponding probabilities for terciles 2 and 3 being p2 and p3 respectively. 
    Obviously, p1 + p2 + p3 = 1.
    If the outcome is in tercile 1 then the forecast is given the LEPS score s = 8/27 p1 – 1/27 p2 – 7/27 p3, 
    if the outcome is  in tercile 2 then the forecast is given the LEPS score s = –1/27 p1 + 2/27 p2 – 1/27 p3, 
    and if the outcome is  in tercile 3 then the forecast is given the LEPS score s = –7/27 p1 – 1/27 p2 + 8/27 p3. 
    These scores range  from –7/27 to +8/27, but can be multiplied by 27/8 to give a scaled LEPS score which ranges from –7/8 to +1. 
    For a sequence {s1,...,sn} of these LEPS scores, calculate two additional sequences, 
    {u1,...,un} which is the sequence of maximum possible LEPS scores given the observed outcomes, and
    {l1,...,ln} which is the sequence of minimum possible LEPS scores given the observed outcomes.
    Obviously this implies that li ≤ si ≤ ui for i = 1,...,n.
    If the outcome for the ith forecast is a tercile 1 or tercile 3 outcome, then ui = 8/27 and li =  –7/27, 
    whereas if the outcome for the ith forecast is a tercile 2 outcome, then ui = 2/27 and li = –1/27.

    The LEPS skill-score for the sequence of forecasts is calculated as
    LEPS skill = (sum (i from 1 to n) s_i) / (sum (i from 1 to n) u_i)) if the mean LEPS score is non-negative, and as 
    LEPS skill = (sum (i from 1 to n) s_i) / (sum (i from 1 to n) l_i)) otherwise. 
    
    The LEPS skill-score can range from –1 to +1.
    The LEPS score base rate for individual forecasts is zero in both the two-category and three-category cases, 
    this being the score awarded to a climatological forecast (p1 = p2 = 1/2 in the two-category case and p1 =  p2 = p3 = 1/3 in the three-category case). 
    The base  rate for the LEPS skill-score is also zero in both cases.

    """

    # According to Fawcett et al. (2005):
    # tercile scoring matrix
    score_matrix = np.array([
    [ 8, -1, -7],   
    [-1,  2, -1],   
    [-7, -1,  8]    
    ]) / 27

    # LEPS score
    s = np.zeros(len(y_obs))

    for i in range(len(y_obs)):
        observed_class = y_obs[i]
        probabilities_i = probabilities[i] 

    s[i] = np.sum(
        probabilities_i * score_matrix[observed_class]
    )
    
    # print (s.shape)
    
    # maximum and minimum possible LPES scores
    u = np.where(
        (y_obs == 0) | (y_obs == 2), # if obs is t1 or t3 
        8/27, # highest for t1, t3
        2/27 # otherwise, highest for t2
    )

    l = np.where(
        (y_obs == 0) | (y_obs == 2), # if obs is t1 or t3
        -7/27, # lowest for t1, t3
        -1/27 # otherwise, lowest for t2
    )
    
    # print (u.shape)
    # print (l.shape)

   # LEPS skill score
    if np.mean(s) >= 0:
       leps_skill = np.sum(s) / np.sum(u)
    else:
       leps_skill = np.sum(s) / np.sum(l)

    return leps_skill



def get_classification_scores(y_prob,y_obs,years_ver):
    """
    ARGS:
        y_prob: probabilities of class numpy array [n,2]: [[t1, t2, t3],...]
        y_obs: true class: [0,1,2,1,...]
        years_ver: years verifed/predicted 
    """
    scores={}

    # Predicted tercile = class with highest probability
    y_pred = np.argmax(y_prob, axis=1) #np.argmax returns index-class (0,1,2) of highest probability

    # Build contingency table
    # yrows = predictions, xcolumns = observations
    contingency = np.zeros((3, 3), dtype=int)
    for pred, obs in zip(y_pred, y_obs):
        contingency[pred, obs] += 1 # adds 1 to according (row, column) 

    scores['contingency'] = contingency

    # # Hit and misses for each tercile (diagonal values in contingency table)
    # scores['hit_t1'] = contingency[0, 0]
    # scores['hit_t2'] = contingency[1, 1]
    # scores['hit_t3'] = contingency[2, 2]

    # scores['miss_t1'] = contingency[1, 0] + contingency[2, 0]
    # scores['miss_t2'] = contingency[0, 1] + contingency[2, 1]
    # scores['miss_t3'] = contingency[0, 2] + contingency[1, 2]

    # Overall accuracy 0 to 1
    # (correct predictions) / (number of predictions) 
    scores['accuracy'] = np.mean(y_pred == y_obs)
    scores['hit_count'] = np.sum(y_pred == y_obs)
    scores['miss_count'] = np.sum(y_pred != y_obs)
    scores['hit_years'] = years_ver[y_pred == y_obs]
    scores['miss_years'] = years_ver[y_pred != y_obs]
    

    # Save observations and probabilities
    scores['class_obs'] = y_obs
    scores['class_pred'] = y_pred
    scores['prob_t1'] = y_prob[:, 0]
    scores['prob_t2'] = y_prob[:, 1]
    scores['prob_t3'] = y_prob[:, 2]

    ## bias = (class predictions) / (class occurrences)
    # bias = 1 predicted frequency matches observed frequency
    # bias < 1 underforecast the tercile
    # bias > 1 overforecast the tercile 
    # bias is 0 to infinity, ignore divide by zero warning

    with np.errstate(divide='ignore', invalid='ignore'): # ignore division by 0, invalid division or operation
        scores['bias_t1'] = np.sum(y_pred == 0) / np.sum(y_obs == 0) # predicted tercile count divided by observation count 
        scores['bias_t2'] = np.sum(y_pred == 1) / np.sum(y_obs == 1)
        scores['bias_t3'] = np.sum(y_pred == 2) / np.sum(y_obs == 2)

    
    scores['LEPS skill']=leps_skill(y_obs,y_prob) ##############change to bom or mine 
    scores['BSS'] = brier_skill_score(y_obs,y_prob)

    
    return scores


# Validation function1: split data into two periods to train/test

def combo_method_SKL_train_test(X1, X2, Y, training_years):

    """ 
    training_years can be range, array
    """

    scores = {}
    
    years = X1.index.year.values
    
    years_tr = training_years
    years_ver = years[years > training_years[-1]]

    prob_roni, prob_oni, q33, q67 = combo_method_SKL_Dine(
        X1=X1,
        X2=X2,
        Y=Y,
        training_years=years_tr,
        prediction_years=years_ver
    )

    # Classify observed validation rainfall using training terciles
    obs_ver = Y[Y.index.isin(years_ver)]

    y_ver = np.ones(len(obs_ver), dtype=int)
    y_ver[obs_ver.values < q33] = 0
    y_ver[obs_ver.values > q67] = 2

    score_roni = get_classification_scores(prob_roni, y_ver, years_ver)
    score_oni = get_classification_scores(prob_oni, y_ver, years_ver)


    scores['roni'] = score_roni
    scores['oni'] = score_oni
    scores['years_training'] = years_tr
    scores['years_verification'] = years_ver
    scores['observed_classes'] = y_ver
    scores['probabilites_roni'] = prob_roni
    scores['probabilites_oni'] = prob_oni

    return scores


# Validation function2:
# change cv = LOOCV, k-fold, sliding windows (timeseriessplit)

def combo_method_SKL_validation(X1,X2,Y,years, 
                                cv=LeaveOneOut()): 
    """
        Cross validate the bom combo method
        ARGS:
            DF: dataframe with roni,oni, columns
            years: list of entire years to validate (or range or ndarray)
            cv: cross validator from sklearn with inherited "split" method
                default = LeaveOneOut() # uses future data 
                other options =  KFold(n_splits=n) # still uses future data
                                 TimeSeriesSplit(n_splits=n, max_train_size =20, test_size=1) # does not use future data          
    """
    if cv is None:
        cv=LeaveOneOut()

    i_sub = X1.index.year.isin(years) # returns True/False 
    X1_sub, X2_sub, Y_sub = X1[i_sub], X2[i_sub], Y[i_sub] 
    if not isinstance(years,np.ndarray): # returns True if an object is an instance of a specified class. Otherwise, it returns False.
        years = np.array(years) # if years are not nparray, make it array
    
    scores=[]
    # i=0
    ## Iterate over dataset using cross validation arg
    for i_train, i_test in cv.split(X1_sub):
        # define training and verification years from entire input years data 
        years_tr = years[i_train]
        years_ver= years[i_test]
        
        # probability from combo method_Dine  
        i_prob_roni, i_prob_oni, q33,q67 = combo_method_SKL_Dine(
            X1=X1_sub,
            X2=X2_sub,
            Y=Y_sub,
            training_years=years_tr, 
            prediction_years=years_ver
        )

        # ## verification based on tercile of training data - done in combo_method_Dine 
        # # use training data to set tercile 
        # obs_tr = Y_sub[i_train]
        obs_ver = Y_sub.values[i_test]
        # q33, q67 = np.quantile(obs_tr, [1/3, 2/3])

        # calculated observed class during verification years using that tercile
        y_ver = np.ones(len(years_ver),dtype=int)
        y_ver[obs_ver < q33] = 0 # obs_ver needs to be observed rainfall data during verification period
        # y_ver[obs_ver.values >= q33] & y_ver[obs_ver.values >= q67] = 1
        y_ver[obs_ver > q67] = 2

        score_roni = get_classification_scores(i_prob_roni, y_ver, years_ver)
        score_oni = get_classification_scores(i_prob_oni, y_ver, years_ver)


        # scores from probability vs observation
        scores.append({
            'years_training': years_tr,
            'years_verification': years_ver,
            'roni': score_roni,
            'oni': score_oni
        })

        # # add some meta data for easier analysis
        # scores[i]['years_training']=years_tr
        # scores[i]['years_verification']=years_ver
        # #scores[i][]
        # i=i+1
    # Calculate some averages
    # validation_means={"LEPS":0, }
    # for score in scores:
    #     mean_leps = mean_leps+score['LEPS skill']
    
    return scores


