# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:26:09 2018

@author: narendran.thesma
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pyramid.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy.random import seed
from fbprophet import Prophet
import progressbar

RawData = pd.read_csv("Final2000Parts.csv")

RawData.ShipRequestDate = pd.to_datetime(RawData.ShipRequestDate)
RawData = RawData.loc[RawData["OrderQuantity"] > 0]
RawData = RawData.dropna(axis=0)
RawData['MonthYear'] = RawData["ShipRequestDate"].apply(lambda dt: dt.replace(day=1))
AggRawData = RawData.groupby(['CardexDiscretePart','MonthYear']).agg({'OrderQuantity':'sum'}).reset_index()
AggRawData['CardexDiscretePart'] = AggRawData['CardexDiscretePart'].str.replace('/', '_')

def ArimaxKLX(trainDF, predDF):
    TrainExogenous = {'Month' : trainDF['MonthYear'].dt.month, 'Year' : trainDF['MonthYear'].dt.year}
    TrainExogenousDF = pd.DataFrame(TrainExogenous)
    TestExogenous = {'Month' : predDF['MonthYear'].dt.month, 'Year' : predDF['MonthYear'].dt.year}
    TestExogenousDF = pd.DataFrame(TestExogenous)
    arimax_fit = auto_arima(trainDF['OrderQuantity'],
                           exogenous=TrainExogenousDF,
                           start_p=0,
                           start_q=0,
                           max_p=6,
                           max_d=2,
                           max_q=6,
                           start_P=0,
                           start_Q=0,
                           max_P=6,
                           max_D=2,
                           max_Q=6,
                           max_order=6,                            
                           seasonal=True,
                           stationary=False,
                           information_criterion='aic',
                           stepwise=False,
                           trace=False,
                           test='adf',
                           seasonal_test='ocsb',                        
                           error_action='ignore',  
                           suppress_warnings=True, 
                           enforce_stationarity=False)
    ArimaxForecast = arimax_fit.predict(len(predDF), exogenous=TestExogenousDF)
    ArimaxForecast = np.round(ArimaxForecast, 0)
    ArimaxForecast = np.clip(ArimaxForecast, 0, np.max(ArimaxForecast))
    return ArimaxForecast

def simpleMovingAverage(trainDF, predDF, order):
    smaForecast = pd.Series(len(predDF))
    trainTS = trainDF["OrderQuantity"]
    for i in range(len(predDF)):
        smaForecast[i] = trainTS.rolling(order).mean().iloc[-1]
        trainTS = trainTS.append(pd.Series(smaForecast[i]))
    return np.array(np.round(smaForecast,0))

def holtsWinters(trainDF, predDF, alpha, beta, gamma, m):
    trainTS = trainDF["OrderQuantity"]
    trainTS.index = trainDF["MonthYear"]
    decomposedTS = seasonal_decompose(trainTS, model='additive', freq=len(predDF))
    decomposedDict = {'OrderQuantity' : decomposedTS.observed, 'Trend' : decomposedTS.trend, 'Seasonality' : decomposedTS.seasonal, 'Error' : decomposedTS.resid}
    decomposedDF = pd.DataFrame(decomposedDict)
    decomposedDF = decomposedDF.fillna(0)
    decomposedDF['Level'] = decomposedDF['OrderQuantity'] - (decomposedDF['Seasonality'] + decomposedDF['Trend'] + decomposedDF['Error'])
    overallDF = decomposedDF.append(predDF)
    yforecast = pd.Series(len(predDF))
    for i in range(len(decomposedDF), len(overallDF)):
        overallDF.Level[i] = alpha * (overallDF.OrderQuantity[i] - overallDF.Seasonality[i-m]) + (1 - alpha) * (overallDF.Level[i-1] +  overallDF.Trend[i-1])
        overallDF.Trend[i] = beta * (overallDF.Level[i] - overallDF.Level[i-1]) + (1 - beta) *  overallDF.Trend[i-1]
        overallDF.Seasonality[i] = gamma * (overallDF.OrderQuantity[i] - overallDF.Level[i - 1] -  overallDF.Trend[i - 1]) + (1 - gamma) * overallDF.Seasonality[i - m]
        hm = ((len(predDF) - 1) % m) + 1
        yforecast[i] =  overallDF.Level[i] + len(predDF) * overallDF.Trend[i] + overallDF.Seasonality[i-m+hm]
        yforecast[i] = round(yforecast[i],0)
        overallDF.OrderQuantity[i] = yforecast[i]
    yforecast = yforecast[1:len(yforecast)]
    yforecast[yforecast <= 0] = 0
    return np.array(yforecast)

def holtWintersForecast(trainDF, predDF, max_alpha, max_beta, max_gamma, m, cv_frame, step = 0.1):
    max_alpha = np.arange(0.0, max_alpha, step)
    max_beta = np.arange(0.0, max_beta, step)
    max_gamma = np.arange(0.0, max_gamma, step)
    TrainedDF = trainDF.copy()
    TrainedModelDF = TrainedDF.head(len(TrainedDF) - cv_frame)
    ValidationModelDF = TrainedDF.tail(cv_frame)
    trainTS = TrainedModelDF["OrderQuantity"]
    trainTS.index = TrainedModelDF["MonthYear"]
    holtForecastCVDF = pd.DataFrame()
    Run = 1
    for i in range(0, len(max_alpha)):
        for j in range(0, len(max_beta)):
            for k in range(0,len(max_gamma)):
                tempHoltForecast = pd.Series()
                tempHoltForecast = holtsWinters(trainDF, ValidationModelDF, max_alpha[i], max_beta[j], max_gamma[k], m)
                #tempHoltForecast.index = ValidationModelDF.index
                tempForecastCVDF = pd.DataFrame()
                tempForecastCVDF['Actuals'] = ValidationModelDF['OrderQuantity']
                tempForecastCVDF = tempForecastCVDF.assign(Forecast = tempHoltForecast)
                tempForecastCVDF["Run"] = Run
                tempForecastCVDF["Alpha"] = max_alpha[i]
                tempForecastCVDF["Beta"] = max_beta[j]
                tempForecastCVDF["Gamma"] = max_gamma[k]
                holtForecastCVDF = holtForecastCVDF.append(tempForecastCVDF)
                Run = Run + 1
    holtForecastCVDF['MSE'] = (holtForecastCVDF['Actuals'] - holtForecastCVDF['Forecast'])**2
    holtForecastCVDF['Forecast'] = round(holtForecastCVDF['Forecast'],0)
    meanMSEHoltsDF = holtForecastCVDF.groupby(['Alpha', 'Beta', 'Gamma', 'Run'])['MSE'].mean().reset_index()
    meanMSEminHoltsDF = meanMSEHoltsDF.loc[meanMSEHoltsDF["MSE"] == np.min(meanMSEHoltsDF["MSE"])]
    meanMSEminHoltsDF = meanMSEminHoltsDF.iloc[0,:]
    holtMSEHoltsDFminRunDF = holtForecastCVDF.loc[holtForecastCVDF['Run'] == meanMSEminHoltsDF['Run']]
    yPredForecast = holtsWinters(trainDF, predDF, holtMSEHoltsDFminRunDF.Alpha.iloc[0], holtMSEHoltsDFminRunDF.Beta.iloc[0], holtMSEHoltsDFminRunDF.Gamma.iloc[0], m)
    return yPredForecast

def lagMatrix(x,lags):
    import pandas as pd
    lagDF = pd.DataFrame()
    for i in lags:
        colName = 'lag_' +  str(i)
        lagDF[colName] = x.shift(i)
    return lagDF

def randomForest(trainDF, predictionDF, lags):
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    predLagDF = lagDF.tail(len(predDF))
    forest_reg = RandomForestRegressor(n_estimators=1000)
    X = trainLagDF.drop("lag_0", axis=1)
    y = trainLagDF["lag_0"]
    forest_reg.fit(X, y)
    predDF['lag_0'] = np.nan
    rfPredDF = np.empty([len(predDF)])
    for i in range(len(predLagDF)):
        tempDF = predLagDF.iloc[[i]]
        predSingleDF = tempDF.drop("lag_0", axis = 1)
        tempDF["lag_0"] = forest_reg.predict(predSingleDF)
        rfPredDF[i] = np.round(tempDF["lag_0"],0)
        trainLagDF = trainLagDF.append(tempDF)
        X = trainLagDF.drop("lag_0", axis = 1)
        y = trainLagDF["lag_0"]
        forest_reg.fit(X, y)
    rfPredDF = rfPredDF.clip(0, np.max(rfPredDF))
    return rfPredDF

def gradientBoosting(trainDF, predictionDF, lags):
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    predLagDF = lagDF.tail(len(predDF))
    gradient_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate = 0.001, min_impurity_split = 0.5, random_state = 42, verbose = 0, subsample = 0.9)
    X = trainLagDF.drop("lag_0", axis=1)
    y = trainLagDF["lag_0"]
    gradient_reg.fit(X, y)
    predDF['lag_0'] = np.nan
    gradPredDF = np.empty([len(predDF)])
    for i in range(len(predLagDF)):
        tempDF = predLagDF.iloc[[i]]
        predSingleDF = tempDF.drop("lag_0", axis = 1)
        tempDF["lag_0"] = gradient_reg.predict(predSingleDF)
        gradPredDF[i] = np.round(tempDF["lag_0"],0)
        trainLagDF = trainLagDF.append(tempDF)
        X = trainLagDF.drop("lag_0", axis = 1)
        y = trainLagDF["lag_0"]
        gradient_reg.fit(X, y)
    gradPredDF = gradPredDF.clip(0, np.max(gradPredDF))
    return gradPredDF

def adaBoosting(trainDF, predictionDF, lags):
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    predLagDF = lagDF.tail(len(predDF))
    adaBoost_reg = AdaBoostRegressor(n_estimators=1000, learning_rate = 0.001,random_state = 42)
    X = trainLagDF.drop("lag_0", axis=1)
    y = trainLagDF["lag_0"]
    adaBoost_reg.fit(X, y)
    predDF['lag_0'] = np.nan
    adaPredDF = np.empty([len(predDF)])
    for i in range(len(predLagDF)):
        tempDF = predLagDF.iloc[[i]]
        predSingleDF = tempDF.drop("lag_0", axis = 1)
        tempDF["lag_0"] = adaBoost_reg.predict(predSingleDF)
        adaPredDF[i] = np.round(tempDF["lag_0"],0)
        trainLagDF = trainLagDF.append(tempDF)
        X = trainLagDF.drop("lag_0", axis = 1)
        y = trainLagDF["lag_0"]
        adaBoost_reg.fit(X, y)
    adaPredDF = adaPredDF.clip(0, np.max(adaPredDF))
    return adaPredDF

def svr(trainDF, predictionDF, lags):
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    predLagDF = lagDF.tail(len(predDF))
    svr_reg = SVR(C = 1.0, epsilon = 0.1, kernel='sigmoid')
    X = trainLagDF.drop("lag_0", axis=1)
    y = trainLagDF["lag_0"]
    svr_reg.fit(X, y)
    predDF['lag_0'] = np.nan
    svrPredDF = np.empty([len(predDF)])
    for i in range(len(predLagDF)):
        tempDF = predLagDF.iloc[[i]]
        predSingleDF = tempDF.drop("lag_0", axis = 1)
        tempDF["lag_0"] = svr_reg.predict(predSingleDF)
        svrPredDF[i] = np.round(tempDF["lag_0"],0)
        trainLagDF = trainLagDF.append(tempDF)
        X = trainLagDF.drop("lag_0", axis = 1)
        y = trainLagDF["lag_0"]
        svr_reg.fit(X, y)
    svrPredDF = svrPredDF.clip(0, np.max(svrPredDF))
    return svrPredDF

def svc(trainingDF, predictionDF, lags):
    trainDF = trainingDF.copy()
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    allDF['OrderQuantity'] = np.where(allDF['OrderQuantity'] > 0, 1.0, 0.0)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    predLagDF = lagDF.tail(len(predDF))
    svc_reg = SVC(C = 1.0, probability = True, kernel='rbf', random_state = 42)
    X = trainLagDF.drop("lag_0", axis=1)
    y = trainLagDF["lag_0"]
    svcPredDF = np.empty([len(predDF)])
    if len(np.unique(y)) == 1:
        svcPredDF = np.repeat(np.unique(y), len(predDF))
    else:
        svc_reg.fit(X, y)
        predDF['lag_0'] = np.nan
        for i in range(len(predLagDF)):
            tempDF = predLagDF.iloc[[i]]
            predSingleDF = tempDF.drop("lag_0", axis = 1)
            tempDF["lag_0"] = svc_reg.predict(predSingleDF)
            svcPredDF[i] = np.round(tempDF["lag_0"],0)
            trainLagDF = trainLagDF.append(tempDF)
            X = trainLagDF.drop("lag_0", axis = 1)
            y = trainLagDF["lag_0"]
            svc_reg.fit(X, y)
        svcPredDF = svcPredDF.clip(0, np.max(svcPredDF))
    return svcPredDF

def prophet(trainDF, predDF):
    prophetTrain = pd.DataFrame()
    prophetTrain["ds"] = trainDF.MonthYear
    prophetTrain["y"] = trainDF.OrderQuantity
    prophet_mdl = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1)
    prophet_mdl.fit(prophetTrain)
    future = prophet_mdl.make_future_dataframe(periods=len(predDF))
    prophetForecast = prophet_mdl.predict(future)
    ProphetForecastPredDF = prophetForecast.tail(len(predDF)) 
    prophetForecastValues = np.array(round(ProphetForecastPredDF['yhat'],0))
    prophetForecastValues = np.clip(prophetForecastValues, 0, np.max(prophetForecastValues))
    prophetForecastValues = np.round(prophetForecastValues, 0)
    return prophetForecastValues

def lstmKLX(trainDF, predictionDF, lags):
    seed(1)
    predDF = predictionDF.copy()
    allDF = trainDF.append(predDF)
    trainDF.index = trainDF["MonthYear"]
    predDF.index = predDF["MonthYear"]
    lagDF = lagMatrix(allDF["OrderQuantity"], lags)
    lagDF.index = allDF["MonthYear"]
    lagDF = lagDF.dropna(axis = 0)
    lagDF['DemadGap'] = lagDF['lag_1'] - lagDF['lag_0']
    lagDF['MonthYear'] = lagDF.index
    lagDF['Month'] = lagDF.MonthYear.dt.month
    lagDF['Year'] = lagDF.MonthYear.dt.year
    lagDF = lagDF.drop("MonthYear", axis = 1)
    trainLagDF = lagDF.drop(predDF.index)
    trainLagDF_X = trainLagDF.drop("lag_0", axis = 1)
    trainLagDF_Y = trainLagDF["lag_0"]
    train_X = trainLagDF_X.head(len(trainLagDF_X) - 3)
    train_y = trainLagDF_Y.head(len(trainLagDF_Y) - 3)
    test_X = trainLagDF_X.tail(3)
    test_y = trainLagDF_Y.tail(3)
    train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(train_X, train_y, epochs=20, batch_size=5, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    predLagDF = lagDF.tail(len(predDF))
    predLagDF_X = predLagDF.drop("lag_0", axis = 1)
    predLagDF_X = predLagDF_X.values.reshape((predLagDF_X.shape[0], 1, predLagDF_X.shape[1]))
    predDF['lag_0'] = np.nan
    lstmPredDF = np.empty([len(predDF)])
    for i in range(len(predLagDF)):
        tempDF = predLagDF.iloc[[i]]
        predSingleDF = tempDF.drop("lag_0", axis = 1)
        predSingleDF = predSingleDF.values.reshape((predSingleDF.shape[0], 1, predSingleDF.shape[1]))
        tempDF["lag_0"] = model.predict(predSingleDF)
        lstmPredDF[i] = np.round(tempDF["lag_0"],0)
        trainLagDF = trainLagDF.append(tempDF)
        trainLagDF_X = trainLagDF.drop("lag_0", axis = 1)
        trainLagDF_Y = trainLagDF["lag_0"]
        train_X = trainLagDF_X.head(len(trainLagDF_X) - 3)
        train_y = trainLagDF_Y.head(len(trainLagDF_Y) - 3)
        test_X = trainLagDF_X.tail(3)
        test_y = trainLagDF_Y.tail(3)
        train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
        model = Sequential()
        model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.fit(train_X, train_y, epochs=20, batch_size=5, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    lstmPredDF = np.clip(lstmPredDF, 0, np.max(lstmPredDF))
    lstmPredDFCopy = lstmPredDF.copy()
    lstmPredDFCopy[lstmPredDFCopy <= 0] = 0
    return lstmPredDFCopy

bar = progressbar.ProgressBar(maxval=len(AggRawData["CardexDiscretePart"].unique()), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
b = 0
#finalForecast = pd.DataFrame()
lags = [0,1,2,3,4,5,6,12,13,23,24]
for i in AggRawData["CardexDiscretePart"].unique():
#for i in range(1):
    try:
        bar.update(b+1)
        b = b + 1
        #i = '03DU03'
        locDF = AggRawData.loc[AggRawData['CardexDiscretePart'] == i]
        minDate =min(locDF['MonthYear'].dt.date)
        maxDate = pd.to_datetime("2017-10-01")
        MonthYear = np.arange(minDate, maxDate, dtype='datetime64[M]')
        CorrectDF = pd.DataFrame(MonthYear)
        CorrectDF.columns = ['MonthYear']
        locDF = CorrectDF.merge(locDF, how="left", on="MonthYear")
        locDF['OrderQuantity'] = locDF['OrderQuantity'].fillna(0)
        locDF['CardexDiscretePart'] = locDF['CardexDiscretePart'].fillna(i)
        trainDF = locDF.loc[locDF['MonthYear'] <= pd.to_datetime('2016-06-01')]
        cvDF = locDF.loc[(locDF['MonthYear'] >= pd.to_datetime('2016-07-01')) & (locDF['MonthYear'] <= pd.to_datetime('2016-09-01'))]
        testDF = locDF.loc[locDF['MonthYear'] >= pd.to_datetime('2016-10-01')]
        cvForecastDF = cvDF.copy()
        errorDF = pd.DataFrame()
        try:
            cvForecastDF["ArimaxForecast"] = ArimaxKLX(trainDF, cvDF)
            cvForecastDF['RMSE_ArimaxForecast'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['ArimaxForecast'])**2)**0.5
            errorDF["meanRMSE_ARIMAX"] = pd.Series(np.mean(cvForecastDF['RMSE_ArimaxForecast']))
        except:
            print('Arimax Failed')
        try:
            cvForecastDF["SMA"] = simpleMovingAverage(trainDF, cvDF, order=3)
            cvForecastDF['RMSE_SMA'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['SMA'])**2)**0.5
            errorDF["meanRMSE_SMA"] = pd.Series(np.mean(cvForecastDF['RMSE_SMA']))
        except:
            print("SMA Failed")
        try:
            cvForecastDF["RandomForest"] = randomForest(trainDF, cvDF, lags)
            cvForecastDF['RMSE_RandomForest'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['RandomForest'])**2)**0.5
            errorDF["meanRMSE_RandomForest"] = pd.Series(np.mean(cvForecastDF['RMSE_RandomForest']))
        except:
            print("RF Failed")
        try:
            cvForecastDF["GradientBoosting"] = gradientBoosting(trainDF, cvDF, lags)
            cvForecastDF['RMSE_GradientBoosting'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['GradientBoosting'])**2)**0.5
            errorDF["meanRMSE_GradientBoosting"] = pd.Series(np.mean(cvForecastDF['RMSE_GradientBoosting']))
        except:
            print("Gradient Boosting Failed")
        try:
            cvForecastDF["AdaBoosting"] = adaBoosting(trainDF, cvDF, lags)
            cvForecastDF['RMSE_AdaBoosting'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['AdaBoosting'])**2)**0.5
            errorDF["meanRMSE_AdaBoosting"] = pd.Series(np.mean(cvForecastDF['RMSE_AdaBoosting']))
        except:
            print("Ada Boosting Falied")
        try:
            cvForecastDF["SVR"] = svr(trainDF, cvDF, lags)
            cvForecastDF['RMSE_SVR'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['SVR'])**2)**0.5
            errorDF["meanRMSE_SVR"] = pd.Series(np.mean(cvForecastDF['RMSE_SVR']))
        except:
            print("SVR Failed")
        #try:
        #    cvForecastDF["LSTM"] = lstmKLX(trainDF, cvDF, lags);
        #    cvForecastDF['RMSE_LSTM'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['LSTM'])**2)**0.5
        #    errorDF["meanRMSE_LSTM"] = pd.Series(np.mean(cvForecastDF['RMSE_LSTM']))
        #except:
        #    print("LSTM Failed")
        try:
            cvForecastDF["Prophet"] = prophet(trainDF, cvDF)
            cvForecastDF['RMSE_Prophet'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['Prophet'])**2)**0.5
            errorDF["meanRMSE_Prophet"] = pd.Series(np.mean(cvForecastDF['RMSE_Prophet']))
        except:
            print("Prophet Failed")
        try:
            cvForecastDF["HoltForecast"] = holtWintersForecast(trainDF, cvDF, 0.2, 0.2, 0.1, 12,3, 0.1)
            cvForecastDF['RMSE_HoltForecast'] = ((cvForecastDF['OrderQuantity'] - cvForecastDF['HoltForecast'])**2)**0.5
            errorDF["meanRMSE_HoltForecast"] = pd.Series(np.mean(cvForecastDF['RMSE_HoltForecast']))
        except:
            print("Holt Exception")

        minModel = errorDF.idxmin(axis=1)[0]

        cvDF.index = cvDF["MonthYear"]
        testDF.index = testDF["MonthYear"]

        trainDF = trainDF.append(cvDF)

        testForecast = testDF.copy()

        if minModel == "meanRMSE_ARIMAX":
            testForecast["Forecast"] = ArimaxKLX(trainDF, testDF)
            testForecast["BestModel"] = "Arimax"
        elif minModel == "meanRMSE_SMA":
            testForecast["Forecast"] = simpleMovingAverage(trainDF, testDF, order=3)
            testForecast["BestModel"] = "SMA"
        elif minModel == "meanRMSE_RandomForest":
            testForecast["Forecast"] = randomForest(trainDF, testDF, lags)
            testForecast["BestModel"] = "RandomForest"
        elif minModel == "meanRMSE_AdaBoosting":
            testForecast["Forecast"] = adaBoosting(trainDF, testDF, lags)
            testForecast["BestModel"] = "AdaBoost"
        elif minModel == "meanRMSE_SVR":
            testForecast["Forecast"] = svr(trainDF, testDF, lags)
            testForecast["BestModel"] = "SVR"
        elif minModel == "meanRMSE_LSTM":
            testForecast["Forecast"] = lstmKLX(trainDF, testDF, lags)
            testForecast["BestModel"] = "LSTM"
        elif minModel == "meanRMSE_Prophet":
            testForecast["Forecast"] = prophet(trainDF, testDF)
            testForecast["BestModel"] = "Prophet"
        elif minModel == "meanRMSE_HoltForecast":
            testForecast["Forecast"] = holtWintersForecast(trainDF, testDF, 0.2, 0.2, 0.1, 12,3, 0.1)
            testForecast["BestModel"] = "Holts"
        elif minModel == "meanRMSE_GradientBoosting":
            testForecast["Forecast"] = gradientBoosting(trainDF, testDF, lags)
            testForecast["BestModel"] = "GradientBoost"
        try:
            testForecast["SaleOrNot"] = svc(trainDF, testDF, lags)
        except:
            testForecast["SaleOrNot"] = svc(trainDF, testDF, [0,1,2])

        testForecast["Forecast_Imputed"] = testForecast["Forecast"] * testForecast["SaleOrNot"]
        
        fileName = "D:\KLX\Oct2016-Sep2017\Python Output\\" +  i + ".csv"
        testForecast.to_csv(fileName)
        
        print("Percentage Completed : ", b*100/len(AggRawData["CardexDiscretePart"].unique()))
        print("Parts Completed : ", b)

        #finalForecast = finalForecast.append(testForecast)
    
    except:
        print("For Loop Exception")
        
bar.finish()