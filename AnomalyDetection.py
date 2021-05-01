import streamlit as st
from datetime import datetime, timedelta
from fbprophet import Prophet  # Faceebok Library
import pandas as pd
from pandas import to_datetime
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
# ARIMA
import warnings
import itertools
import statsmodels.api as sm

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

# plt.rcParams['figure.figsize'] = (16.0, 12.0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


def AnomalyDetection():

    method = st.sidebar.selectbox(
        "select a method", ('Simple', 'SARIMA', 'Prophet - Facebook'))

    if method == "Simple":
        st.write(
            """### **Simple Model**""")
        wind = st.sidebar.number_input(
            "Window:", min_value=5, value=15, step=1)
        sigma = st.sidebar.number_input(
            "xSigma:", min_value=0.50, value=2.00, step=0.01)

    elif method == "Prophet - Facebook":
        UncInt = st.sidebar.slider(
            "select the confidence intervals", 0.50, 1.00, 0.95, 0.01)
        st.sidebar.write("confidence intervals: ", UncInt)

        st.write("""### **Prophet - Facebook Model**""")
    elif method == "SARIMA":
        st.write(
            """### **Seasonal Auto Regressive Integrated Moving Average (SARIMA) Model**""")

    st.sidebar.info("""
        [More information](http://gonzalezmaw.pythonanywhere.com/)
        """)

    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:

        # filePath = __file__
        # t.write(filePath)
        df = pd.read_csv(uploaded_file)
        # df2 = pd.read_csv(uploaded_file, parse_dates=['date'], index_col='date'))

        df = df.dropna()
        df = df.drop_duplicates()

        col1, col2 = st.beta_columns(2)

        with col1:
            st.write("Shape of dataset:")
            st.info(df.shape)
        with col2:
            st.write("Complete data:")
            st.info(len(df))

        dataset = st.checkbox("show Dataset")
        if dataset:
            st.write(df)

        col1, col2 = st.beta_columns(2)

        with col1:
            columsList = df.columns.tolist()

            columSelectX = st.selectbox(
                "Select Date columns for the analysis", columsList)

        with col2:
            index = columsList.index(columSelectX)
            select = np.delete(columsList, index)
            columSelectY = st.selectbox(
                "Select the Target for the analysis", select)

        if len(columSelectX) > 0 and columSelectY != None:

            X = df[columSelectX]
            y = df[columSelectY]  # Selecting the last column as Y

            df2 = pd.DataFrame()
            df2[columSelectX] = df[columSelectX]
            df2[columSelectY] = df[columSelectY]
            df2[columSelectX] = to_datetime(df[columSelectX])
            # st.write(df2)

            df2 = df2.sort_values(
                by=columSelectX, ascending=True)

            dfChart = pd.DataFrame()
            dfChart[columSelectX] = to_datetime(df2[columSelectX])
            dfChart[columSelectY] = df[columSelectY]

            # set date as index
            dfChart.set_index(columSelectX, inplace=True)
            dfChart.index
            # st.write(dfChart)
            #
            y_ = dfChart[columSelectY].resample('MS').mean()
            # y = dfChart[columSelectY].resample('w').mean()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        ax = dfChart.plot()
        ax.grid(b=True, which='major', color='lightgrey', linestyle='-')

        st.pyplot()

        # dfChart.hist()

        if method == "Simple":
            # st.write(dfChart.std())

            dfChart["lower"] = dfChart[columSelectY].rolling(window=wind)\
                .mean() - (sigma * dfChart[columSelectY].rolling(window=wind).std())
            dfChart["upper"] = dfChart[columSelectY].rolling(window=wind)\
                .mean() + (sigma * dfChart[columSelectY].rolling(window=wind).std())

            dfChart.plot()
            st.pyplot()

            dfChart["anomaly"] = dfChart.apply(lambda row: row[columSelectY] if (
                row[columSelectY] <= row["lower"] or row[columSelectY] >= row["upper"]) else 0, axis=1)

            ax = dfChart[['lower', 'upper', 'anomaly']].plot(
                figsize=(12, 8), color=['darkorange', 'green', 'red'])
            ax.grid(b=True, which='major', color='lightgrey', linestyle='-')

            st.pyplot()

            dfChart["anomaly_"] = dfChart.apply(
                lambda row: -1 if (row[columSelectY] <= row["lower"]) else (1 if (row[columSelectY] >= row["upper"]) else 0), axis=1)

            # or row[columSelectY] >= row["upper"]) else 0, axis=1)

            # dfChart["anomaly2"]=dfChart.apply(lambda x : x*2 if x < 10 else (x*3 if x < 20 else x)

            ax = dfChart[['anomaly']].plot(figsize=(12, 8), color=['red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            ax = dfChart[['anomaly_']].plot(figsize=(12, 8), color=['red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            st.write(dfChart)

        elif method == "Prophet - Facebook":

            # prepare expected column names
            df2.columns = ['ds', 'y']
            df2['ds'] = to_datetime(df2['ds'])
            # define the model
            model = Prophet()
            # fit the model

            model = Prophet(interval_width=UncInt)

            model.fit(df2)
            st.subheader("""**Training**""")
            future = list()
            future = df2.iloc[:, 0]
            future = DataFrame(future)
            # future.columns = ['ds']
            # future['ds'] = to_datetime(future['ds'])
            forecast = model.predict(future)
            # st.write(forecast)

            # forecast = model.predict(df2.iloc[:, 0])
            # summarize the forecast
            # st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # plot forecast
            fig, ax = plt.subplots()
            ax = model.plot(forecast)
            # st.write(ax)
            plt.xlabel('Date', fontsize=17)
            plt.ylabel(columSelectY, fontsize=17)
            plt.title('Time Series - Training', fontsize=20)
            st.pyplot()

            # calculate MAE between expected and predicted values for december
            y_true = df2['y'].values
            y_pred = forecast['yhat'].values

            col1, col2 = st.beta_columns(2)

            with col1:
                mae = mean_absolute_error(y_true, y_pred)
                st.success('Mean Absolute Error: %.4f' % mae)
            with col2:
                mape = mean_absolute_percentage_error(y_true, y_pred)
                st.success('Mean Absolute Percentage Error: %.3f' % mape)

            # plot expected vs actual
            # st.write(df2)
            dfPredChart = DataFrame()
            dfPredChart[columSelectX] = df2['ds']
            dfPredChart[columSelectY] = df2['y'].values
            dfPredChart['Predicted'] = forecast['yhat'].values
            dfPredChart['yhat_lower'] = forecast['yhat_lower'].values
            dfPredChart['yhat_upper'] = forecast['yhat_upper'].values

            dfPredChart.set_index(columSelectX, inplace=True)
            ax = dfPredChart[[columSelectY, 'Predicted']].plot(figsize=(12, 8))
            ax.grid(b=True, which='major', color='lightgrey', linestyle='-')
            st.pyplot()

            dfPredChart["anomaly"] = dfPredChart.apply(lambda row: row[columSelectY] if (
                row[columSelectY] <= row["yhat_lower"] or row[columSelectY] >= row["yhat_upper"]) else 0, axis=1)

            dfPredChart["anomaly_"] = dfPredChart.apply(
                lambda row: -1 if (row[columSelectY] <= row["yhat_lower"]) else (1 if (row[columSelectY] >= row["yhat_upper"]) else 0), axis=1)

            ax = dfPredChart[['yhat_lower', 'yhat_upper', 'anomaly']].plot(
                figsize=(12, 8), color=['darkorange', 'green', 'red'])
            ax.grid(b=True, which='major', color='lightgrey', linestyle='-')

            st.pyplot()

            ax = dfPredChart[['anomaly_']].plot(figsize=(12, 8), color=['red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            st.write(dfPredChart)

            # st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        elif method == "SARIMA":

            st.subheader("""**Training**""")
            # Define the p, d and q parameters to take any value between 0 an.d 2
            p = d = q = range(0, 2)

            # Generate all different combinations of p, q and q triplets
            pdq = list(itertools.product(p, d, q))

            # Generate all different combinations of seasonal p, q and q triplets
            seasonal_pdq = [(x[0], x[1], x[2], 12)
                            for x in list(itertools.product(p, d, q))]

            # specify to ignore warning messages
            warnings.filterwarnings("ignore")

            paramList = list()
            aicList = list()
            param_seasonalList = list()

            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    # try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    paramList.append([param])

                    param_seasonalList.append([param_seasonal])
                    aicList.append([results.aic])

            df3 = pd.DataFrame(paramList)
            df3["Seasonal Component"] = pd.DataFrame(param_seasonalList)
            df3["AIC"] = pd.DataFrame(aicList)
            df3.columns = ['Parameters', 'Seasonal Component', 'AIC']

            df3_New = df3.sort_values(
                by='AIC', ascending=True)

            # st.write(df3_New)

            model = sm.tsa.statespace.SARIMAX(y,
                                              order=df3_New.iloc[0, 0],
                                              seasonal_order=df3_New.iloc[0, 1],
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)

            results = model.fit()

            summary = pd.DataFrame(results.summary().tables[1])

            # st.write(summary)
            results.plot_diagnostics(figsize=(16, 8))

            diagnostic = st.checkbox("Show Diagnostic Charts")
            if diagnostic:
                results.plot_diagnostics(figsize=(15, 12))
                st.pyplot()

            df['Predicted'] = results.predict(
                start=0, end=len(df2)-1, dynamic=False)

            pred = results.get_prediction(
                start=0, end=len(df2)-1, dynamic=False)
            pred_ci = pred.conf_int()
            df['lower'] = pred_ci.iloc[:, 0]
            df['upper'] = pred_ci.iloc[:, 1]

            df[columSelectX] = to_datetime(df[columSelectX])
            # st.write(df)

            # calculate MAE between expected and predicted values for december
            y_true = df[columSelectY].values
            y_pred = df['Predicted'].values

            col1, col2 = st.beta_columns(2)

            with col1:
                mae = mean_absolute_error(y_true, y_pred)
                st.success('Mean Absolute Error: %.4f' % mae)
            with col2:
                mape = mean_absolute_percentage_error(y_true, y_pred)
                st.success('Mean Absolute Percentage Error: %.3f' % mape)

            # Get the chart in date
            dfPredChart = DataFrame()
            dfPredChart[columSelectX] = to_datetime(df[columSelectX])
            dfPredChart[columSelectY] = (df[columSelectY])
            dfPredChart['Predicted'] = results.predict(
                start=0, end=len(df2)-1, dynamic=False)
            dfPredChart['lower'] = df['lower']
            dfPredChart['upper'] = df['upper']

            dfPredChart["anomaly"] = dfPredChart.apply(lambda row: row[columSelectY] if (
                row[columSelectY] <= row["lower"] or row[columSelectY] >= row["upper"]) else 0, axis=1)

            dfPredChart["anomaly_"] = dfPredChart.apply(
                lambda row: -1 if (row[columSelectY] <= row["lower"]) else (1 if (row[columSelectY] >= row["upper"]) else 0), axis=1)

            dfPredChart.set_index(columSelectX, inplace=True)
            ax = dfPredChart[[columSelectY, 'Predicted']].plot(figsize=(12, 8))
            plt.title('Time Series - Training', fontsize=20)
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            ax = dfPredChart[['lower', 'upper', 'anomaly']].plot(
                figsize=(12, 8), color=['darkorange', 'green', 'red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            ax = dfPredChart[['anomaly']].plot(figsize=(12, 8), color=['red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            ax = dfPredChart[['anomaly_']].plot(figsize=(12, 8), color=['red'])
            ax.grid(b=True, which='major', color='lightgray', linestyle='-')
            st.pyplot()

            st.write(dfPredChart)

            # st.write(df2)

    else:
        st.info('Awaiting for CSV file to be uploaded.')
    return
