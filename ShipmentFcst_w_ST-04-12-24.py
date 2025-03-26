#import itertools
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#from math import sqrt
#import seaborn as sns
#from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
#import matplotlib.pyplot as plt
#from datetime import datetime


import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing, Holt
import pandas as pd
import pmdarima as pm
from sklearn import model_selection
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
import warnings
warnings.filterwarnings('ignore')

# Reading data sets
path = "C:/PRINT_COE/Project/STCI_IBP_Data_Analysis/ImpFilesOnSTCI_Fcst"
file_name = "/HW_PLMKT_Level_MasterFile.xlsm"
#file_name = "/SAP-ML-HPS_HW_MasterFile.xlsm"
#file_name = "/SAP-ML(SKU-Mkt).xlsm"
#sheet="Worksheet1"
#sheet="Accuracy_Check"
sheet = "PL_Mkt"
nrow = 900      # number of rows to read

############################# Parameter Defining #################################
hist_periods = 36
to_fcst = 23
to_skiprows = 3                   # if header, skip
hist2fcst = "True_Orders_(TSD)"

#Volatility % to compress
to_compress = 5

###############################################################################
df = pd.read_excel(path + file_name, 
                   sheet_name = sheet, 
                   #usecols = "H:FP",
                   usecols = "H:BP",
                   #usecols = "H:AT",
                   skiprows = to_skiprows,
                   nrows = nrow)

# Making unique key column
con_cat = df.iloc[:,0].map(str) + '_' + df.iloc[:,1].map(str)
df.insert(loc = 2, column = "Key", value = con_cat)

##########################################################################################################################
def fwd_season(season, fcst): 
    fit_extend = season.tshift(len(out_sample))
    fwd_only = fit_extend.tail(len(out_sample))
    ft = fcst + fwd_only
    return ft

def create_features(xg_df):
    xg_df['Month'] = xg_df.index.month
    xg_df['DaysofMonth'] = xg_df.index.daysinmonth
    xg_df['Quarter'] = xg_df.index.quarter
    xg_df['Year'] = xg_df.index.year
    xg_df['DayOfYear'] = xg_df.index.dayofyear
    xg_df['WeekOfYear'] = xg_df.index.isocalendar().week
    return xg_df 

def data_cleaning(uv_ts):
    
    #val = next(index for index,value in enumerate(uv_ts) if value > 0)
    #uv_start_val = uv_ts[val: ]
    #uv_start_val = uv_ts.head(hist_periods)
    #uv_hist = uv_start_val.copy()
    
    nulls = uv_ts.mask(uv_ts < 1)        
    uv_data = nulls.fillna(value = nulls.mean())
    
    # Seasonal Decomposition
    stl = STL(uv_data, period = 12)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid

    # Anomalies treatment from residual components
    lower_resid = np.percentile(resid, to_compress)          
    upper_resid = np.percentile(resid, (100 - to_compress))

    resid = resid.copy()
    resid[resid < lower_resid] = lower_resid
    resid[resid > upper_resid] = upper_resid

    # From Seasonal components 
    lower_season = np.percentile(seasonal, to_compress)          
    upper_season = np.percentile(seasonal, (100 - to_compress)) 

    season = seasonal.copy()
    season[season < lower_season] = lower_season
    season[season > upper_season] = upper_season

    # Combining Cleaned components
    season = season.groupby(season.index.month).transform('mean')
    trend_season_resid = trend + season + resid
    
    tsr = trend_season_resid.mask(trend_season_resid < 1)
    tsr = tsr.bfill()
    tsr = tsr.ffill()
    
    tes_data = pd.Series(data = tsr, index = pd.DatetimeIndex(tsr.index, freq ='MS'))
    
    
    return(tes_data)


params = {
    "bootstrap"         : [True, False],
    "criterion"         : ['mse', 'mae'],
    "learning_rate"     : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth"         : [2, 5, 8, 11, 14, 17, 20],
    "min_child_weight"  : [2, 4, 6, 8],
    "gamma"             : [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree"  : [0.3, 0.5, 0.7, 0.9],
    "colsample_bylevel" : [0.3, 0.5, 0.7, 0.9],
    "n_estimators"      : [50, 100, 200, 300, 500]    
}


def grid_cv(x):
    param_grid={
        'bootstrap'         : [x.best_params_['bootstrap']],
        "criterion"         : [x.best_params_['criterion']],
        "learning_rate"     : [x.best_params_['learning_rate']],
        "max_depth"         : [x.best_params_['max_depth']-1,
                               x.best_params_['max_depth'],
                               x.best_params_['max_depth']+1],
        "min_child_weight"  : [x.best_params_[ "min_child_weight"]-1, 
                               x.best_params_[ "min_child_weight"],
                               x.best_params_[ "min_child_weight"]+1],
        "gamma"             : [x.best_params_[ "gamma"]],
        "colsample_bytree"  : [x.best_params_[ "colsample_bytree"]-0.1, 
                               x.best_params_[ "colsample_bytree"],
                               x.best_params_[ "colsample_bytree"]+0.1],
        "colsample_bylevel" : [x.best_params_[ "colsample_bylevel"]-0.1, 
                               x.best_params_[ "colsample_bylevel"],
                               x.best_params_[ "colsample_bylevel"]+0.1],
        "n_estimators"      : [x.best_params_[ "n_estimators"]-30,
                               x.best_params_[ "n_estimators"],
                               x.best_params_[ "n_estimators"]+30]
        }
    
    return(param_grid)


############################################### Sell-Thru Forecasting ####################################################
############################################################################################################################

key_dim = df.iloc[:,2].unique()

key_list = 0
df_list = []

for i in key_dim: 
    
    key_list = df[df.iloc[:,2]==i] 
    key_list = key_list.iloc[:,3:]
    key_list = key_list.set_index(key_list.iloc[:,0])
    key_list = key_list.iloc[:, 1:].T
    key_list.columns = key_list.columns.str.replace(' ','_')
    key_list = key_list.set_index(pd.to_datetime(key_list.index, format = '%y-%b'))


    uv_ts = key_list[hist2fcst].head(hist_periods)
    val = next(index for index,value in enumerate(uv_ts) if value > 0)
    uv_start_val = uv_ts[val: ]
    
    nulls = uv_ts.mask(uv_ts < 1)        
    uv_data = nulls.fillna(value = 1)

    # Model selection criteria
    Yr_Avg = uv_ts.fillna(value = 0).resample('12M', origin = 'start', closed = 'left').mean()
    avg_st = nulls.tail(6).sum()/6
    count_6 = nulls.tail(6).count()
    count = uv_start_val.count()
    cols = key_list.columns
    
    
    if (uv_ts.tail(12).sum()==0):                #Exclude if no shipmet in 12 months
        continue 
        
    out_sample = pd.date_range(start= max(uv_ts.index), periods= to_fcst, freq = '1M') + pd.offsets.MonthBegin(1)
        
    ############################################### ST Seasonal component Forecast ###########################################  

    if key_list["ST"].sum()>0:
        corrected_st = data_cleaning(key_list["ST"].head(hist_periods))
    else:
        corrected_st = key_list["ST"].head(hist_periods)
    #corrected_st = key_list["ST"].head(hist_periods)
    #corrected_st = corrected_st.fillna(value = corrected_st.mean())
    
    corrected_ship = data_cleaning(key_list[hist2fcst].head(hist_periods))
    #corrected_ship = key_list[hist2fcst].head(hist_periods)
    #corrected_ship = corrected_ship.fillna(value = corrected_ship.mean())
    
    
    FEATURES = ['MS','Lag1MS', 'Lag2MS', 'Lag3MS','Month', 'DaysofMonth', 'Quarter', 'Year', 'DayOfYear', 'WeekOfYear']
    TARGET = "ST"

    kf = KFold(n_splits=10, shuffle=False)
    regressor = xgb.XGBRegressor()
        
    model = model_selection.RandomizedSearchCV(estimator = regressor, param_distributions= params, random_state=42,
                                               n_iter=5, scoring= 'neg_mean_absolute_error', n_jobs= -1, cv=kf)
    
    xgb_ms = key_list[['MS']]
    xgb_ms['Lag1MS'] = xgb_ms['MS'].shift(-1)
    xgb_ms['Lag2MS'] = xgb_ms['MS'].shift(-2)
    xgb_ms['Lag3MS'] = xgb_ms['MS'].shift(-3)
    
    xgb_ms = xgb_ms.fillna(value = xgb_ms.mean())

    xgb_h = pd.concat([corrected_st, xgb_ms], axis=1)
    xgb_h.columns = ["ST", "MS", 'Lag1MS', 'Lag2MS', 'Lag3MS']
      

########################### Sell-Thru forecasting Using MS and its lags ###############################

    if (Yr_Avg[2] > Yr_Avg[1] and Yr_Avg[2] > Yr_Avg[0]) or xgb_ms['MS'].sum()==0:
        
        
        st_drivers = xgb_h[["MS", 'Lag1MS', 'Lag2MS', 'Lag3MS']]

        sxmodel = pm.auto_arima(corrected_st, 
                        exogenous=st_drivers.head(hist_periods),
                        start_p=0, start_q=0,
                        test='adf',
                        max_p=1, max_q=1, 
                        m=12,
                        d=0, D=0, 
                        trace=False,
                        seasonal=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=False)

        output = sxmodel.predict(n_periods=to_fcst,
                                 exogenous=st_drivers.tail(to_fcst))

        tes_model = ExponentialSmoothing(corrected_st, 
                                         trend = "add", 
                                         seasonal = "add", 
                                         seasonal_periods = 12,
                                         damped_trend = True,
                                         initialization_method='estimated').fit()

        tes_output = tes_model.forecast(len(out_sample))
        combo_fcst = pd.Series((output+tes_output)/2, index=out_sample)

        print("Classical fcst for ST :" + i)

    
    else:

        xgb_hist = create_features(xgb_h)
        xgb_f = xgb_hist.head(hist_periods)

        x_all = xgb_f[FEATURES]
        y_all = xgb_f[TARGET]

        model_randCV = model.fit(x_all, y_all)
        param_grid = grid_cv(model_randCV)
        grid_search=GridSearchCV(estimator = regressor, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

        best_algo = grid_search.fit(x_all, y_all)
        xgb_out_s = xgb_hist.tail(to_fcst)
        xgb_fcst = xgb_out_s[FEATURES]
        xgb_output = best_algo.predict(xgb_fcst)
        
        combo_fcst = pd.Series(xgb_output,index=out_sample)
        
        print("XGBoost Fcst for ST:" + i)
    
    
##################################### Shipment Forecast #######################################    

        ST_w_Fcst = pd.concat([corrected_st, combo_fcst], axis=0) 

        frames = [corrected_ship, xgb_ms, ST_w_Fcst]
        new_df = pd.concat(frames, axis=1)
        new_df.columns = ['Ship-Actl_PGI', 'MS', 'Lag1MS', 'Lag2MS', 'Lag3MS', 'ST'] 

        new_df['lag1ST'] = new_df['ST'].shift(-1)
        new_df['lag2ST'] = new_df['ST'].shift(-2)
        new_df['lag3ST'] = new_df['ST'].shift(-3)

        new_df = new_df.ffill()


    if (Yr_Avg[2] > Yr_Avg[1] and Yr_Avg[2] > Yr_Avg[0]) or xgb_ms['MS'].sum()==0 or corrected_st.sum()==0:
        

            shipment_drivers = new_df[["ST", 'lag1ST', 'lag2ST', 'lag3ST', 'MS']]
            
            sxmodel = pm.auto_arima(corrected_ship, 
                            exogenous=shipment_drivers.head(hist_periods),
                            start_p=0, start_q=0,
                            test='adf',
                            max_p=1, max_q=1, 
                            m=12,
                            d=0, D=0, 
                            trace=False,
                            seasonal=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=False)

            ship_output = sxmodel.predict(n_periods=to_fcst,
                                     exogenous=shipment_drivers.tail(to_fcst))
            

            tes_model = ExponentialSmoothing(corrected_ship, 
                                             trend = "add", 
                                             seasonal = "add", 
                                             seasonal_periods = 12,
                                             damped_trend = True,
                                             initialization_method='estimated').fit()

            tes_output = tes_model.forecast(len(out_sample))
            
            ship_fcst = pd.Series((ship_output+tes_output)/2, index=out_sample)
            
            print("Classical fcst for Shipment: " + i)
             
    else:

        FEATURES1 = ['ST', 'lag1ST', 'lag2ST', 'lag3ST', 'Month', 'DaysofMonth', 'Quarter', 'DayOfYear', 'WeekOfYear']   
        TARGET1 = "Ship-Actl_PGI"
        actl_df = create_features(new_df)
        actl_hist = actl_df.head(hist_periods)
        actl_hist = actl_hist.fillna(value=1)

        x_all1 = actl_hist.head(hist_periods)[FEATURES1]
        y_all1 = actl_hist.head(hist_periods)[TARGET1]

        model_randCV1 = model.fit(x_all1, y_all1)
        param_grid = grid_cv(model_randCV1)
        grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, cv=5, n_jobs=-1, verbose=2)

        best_algo1 = grid_search.fit(x_all1, y_all1)

        actl_f = actl_df.tail(to_fcst)
        actl_fcst = actl_f[FEATURES1]
        actl_fcst = actl_fcst.ffill()
        
        ship_fcst = pd.Series(best_algo1.predict(actl_fcst), index=actl_fcst.index)

        
        print('XGBoost Fcst for Shipment:' + i)
    
    
########################################Dataframes for exporting the forecast########################################
    
    
    fit_df = pd.DataFrame(ship_fcst, index = out_sample)

    masked = fit_df.mask(fit_df < 0)
    masked_df = masked.ffill()
    masked_df = np.ceil(masked_df.bfill())
    
    
    for_plting = masked_df.copy()
    for_plting = pd.concat([for_plting, combo_fcst], axis = 1)
    for_plting.columns = ["ShipFcst w ST", "Sell-Thru Fcst"]

    masked_df.index = masked_df.index.strftime('%y-%b')
    data_set = masked_df.T
    data_set.insert(loc = 0, column = "Dim-2", value = i)
    data_set.insert(loc = 1, column = "Key Figure", value = ["ShipFcst w ST"])
    df_list.append(data_set)

    #print(mean_absolute_percentage_error(key_list['Ship-Actl_PGI'].head(hist_periods+(36-hist_periods)).tail(3), fit_df.head(3)))
    
    
printables = pd.concat(df_list, axis = 0)
printables[['Dim-2', 'Dim-1']] = printables['Dim-2'].str.split('_', expand=True)
printables.insert(1, 'Dim-1', printables.pop("Dim-1"))

printables.to_csv(path + "/ML_OutPut/HPS_OPS_HW.csv", index = False)
