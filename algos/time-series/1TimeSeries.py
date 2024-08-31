import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

import time
import numpy as np
from pmdarima import auto_arima

st.set_page_config(layout='wide', page_title='Time Series Analysis', page_icon=':mostly_sunny:')

st.header("Time Series Analysis")

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('./AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Data', 'Decomposing the TS','ARIMA basics','Find Best Model', 'Fit Model', 'Evaluate the model', 'Forecast'])

with tab1:
    st.write(df.head(10))
    df['Passengers'].plot()
    
    st.pyplot()
    st.write(df.dtypes)
    
with tab2:
    st.write('To decompose the time series, we need to import the following package')
    st.code('from statsmodels.tsa.seasonal import seasonal_decompose')
    st.write('And with the following commands we slice up the time series in its components and plot them')
    st.code("result = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)")
    st.code("result.plot()")
    if st.button('Decompose the time series'):
        #decompose the time series
        result = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
        # result = seasonal_decompose(x=df['#Passengers'], model='multiplicative',extrapolate_trend='freq', period=12)
        
        result.plot()
        st.pyplot()

with tab3:
    st.header("ARIMA Basics")
    st.markdown("#### :red[AR]:green[I]:blue[MA] stands for autoregressive integrated moving average model and is specified by three order parameters: (p, d, q).",unsafe_allow_html=True)
    st.markdown("* :red[AR(p) Autoregression] - Captures Seasonality – a regression model that utilizes the dependent relationship between a current observation and observations over a previous period. An auto regressive (AR(p)) component refers to the use of <span style='color:orange; font-weight: bolder'>past values</span> in the regression equation for the time series.\n * :green[I(d) Integration] - Captures Trend – uses differencing of observations (subtracting an observation from observation at the previous time step) in order to make the time series stationary. Differencing involves the subtraction of the current values of a series with its previous values <span style='color:lightgreen; font-weight: bolder'>d number of times</span>.\n\n * :blue[MA(q) Moving Average] - Captures remaining patterns – a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. A moving average component depicts the error of the model as a combination of previous error terms. The order <span style='color:cyan; font-weight: bolder'>q represents the number of terms to be included in the model</span>. ",unsafe_allow_html=True)
    
    st.markdown("#### The ‘auto_arima’ function from the ‘pmdarima’ library helps us to identify the most optimal parameters for an ARIMA model and returns a fitted ARIMA model.")
    st.markdown("#### There are three types of ARIMA. \n 1. ARIMA (Non-seasonal AR) \n 2. SARIMA (Seasonal AR) \n 3. SARIMAX (Seasonal AR with exogenous variables)")
    st.markdown("#### Auto-arima will fit the model in step-wise fashion and will suggest the most optimal parameters and type for an ARIMA model")

with tab4:
    st.subheader('Let us import the library')
    st.code('from pmdarima import auto_arima')
    # Ignore harmless warnings
    import warnings
    warnings.filterwarnings("ignore")

    st.code('Now we can run the step-wise ARIMA, with the following code:\n')

    st.code("# Fit auto_arima function to AirPassengers dataset \n stepwise_fit = auto_arima(df['#Passengers'], start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True, d = None, D = 1, trace = True, \n                           error_action ='ignore', # we don't want to know if an order does not work \n                           suppress_warnings = True, # we don't want convergence warnings \n                          stepwise = True)# set to stepwise")
    
    st.write("For more details on the model, you can visit https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html")
    
    if st.button("Stepwise fit"):
        # Fit auto_arima function to AirPassengers dataset
        stepwise_fit = auto_arima(df['Passengers'], start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True, d = 1, D = 1, trace = True,
                          error_action ='ignore', # we don't want to know if an order does not work 
                          suppress_warnings = True, # we don't want convergence warnings 
                          stepwise = True)# set to stepwise
        
        st.write(stepwise_fit.summary())

        st.code("The stepwise fit found SARIMAX to be the optimum model here. That means the series is affected by exogenous variables, and highly seasonal.")

    
with tab5:
    st.markdown("## Let's fit the SARIMAX model")
    st.write("We will split the data in train and test datasets")
    st.code("# Let's split the data in training and testing sets \n train = df.iloc[:len(airline)-12] # Train data contains 11 out of total 12 years data \n test = df.iloc[len(airline)-12:] # test data is of 1 year")
    train = df.iloc[:len(df)-12]
    test = df.iloc[len(df)-12:]
    st.markdown("##### Further, we need to import the relevant libraries, and fit the model")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    st.code("from statsmodels.tsa.statespace.sarimax import SARIMAX \n model = SARIMAX(train['Passengers'], order = (0, 1, 1), seasonal_order =(2, 1, 1, 12)) \n #seasonal_order parameters are explained below:\n 1. 'P': How did things behave in the previous year at this time of the year? P = 2 means model considers data from 2 months ago. \n 2. 'D': Order of Difference to remove seasonality. D = 1 means that each data point in the seasonal component is replaced with the difference between that data point and the data point from one season ago. \n 3. 'Q' : The order of moving average component. Q = 1 implies that you're using the error from the previous season to adjust your predictions for the current season in the seasonal component of the time series. \n 4. 's' : The interval. For monthly data 's' will assume value 12")
    #if st.button("Fit Model"):
    model = SARIMAX(train['Passengers'], order = (0, 1, 1), seasonal_order =(2, 1, 1, 12))
    result = model.fit()
    
    # st.write(result.summary())
    st.write("\n")
    st.markdown("To model this data using SARIMAX, we specify the non-seasonal component as SARIMAX(p=0, d=1, q=1), which means we use the current month's passengers, the first difference of the passengers, and the previous month's forecast error to make our predictions. \n For the seasonal component, we might specify a seasonal_order of (P=2, D=1, Q=1, m=12), which means we use the number of passengers from 12 months ago, the first difference of the number of passengers over 12 months, and the forecast error from 12 months ago to capture the seasonal patterns in the data.")

with tab6:
    st.subheader("Evaluate the model")
    st.write("For this we will predict the values for test data and plot them on one graph.")
    start = len(train)
    end = len(train) + len(test) - 1

    # Predictions for one-year against the test set
    predictions = result.predict(start, end, type = 'levels').rename("Predictions")

    # plot predictions and actual values
    predictions.plot(legend = True)
    test['Passengers'].plot(legend = True)
    st.pyplot()
    mean_squared_error(test["Passengers"], predictions)
    rmseError = rmse(test["Passengers"], predictions)
    st.write("RMSE : ", rmseError)

with tab7:
    st.subheader("To forecast the values, we will train the model on the complete dataset")
    # Train the model on the full dataset
    model = model = SARIMAX(df['Passengers'], order = (0, 1, 1), seasonal_order =(2, 1, 1, 12))
    result = model.fit()

    # Forecast for the next 3 years
    forecast = result.predict(start = len(df), end = (len(df)-1) + 3 * 12, type = 'levels').rename('Forecast')

    # Plot the forecast values
    df['Passengers'].plot(figsize = (12, 5), legend = True)
    forecast.plot(legend = True)
    st.pyplot()